import os,argparse,subprocess,json,logging,glob,random,uuid,sys
from typing import List,Dict,Optional,Any
from functools import wraps
from dataclasses import dataclass

LOG_TO_DISK=False
LOG_TO_CONSOLE=True

@dataclass
class TTSConfig:
    model_name:str
    language_idx:str
    device:str
    vocoder_name:Optional[str]=None
    speaker_wav:Optional[List[str]]=None
    use_cuda:bool=False
    out_path:str='tts_output.wav'
    source_wav:Optional[str]=None
    target_wav:Optional[str]=None

def setup_logging():
    handlers=[]
    if LOG_TO_CONSOLE:handlers.append(logging.StreamHandler(sys.stdout))
    if LOG_TO_DISK:handlers.append(logging.FileHandler('app.log'))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',handlers=handlers)
    sys.stdout.flush()

def check_tts_command_exists()->bool:
    try:
        result=subprocess.run(['which','tts'],capture_output=True,text=True,check=True)
        logging.info(f"Subprocess flags for 'which tts': {result.args}")
        return True
    except subprocess.CalledProcessError:
        logging.error("tts command not found.")
        return False

def get_speaker_wav(input_wav_dir:str)->List[str]:
    wav_files=glob.glob(os.path.join(input_wav_dir,"*.wav"))
    random.shuffle(wav_files)
    logging.info(f"Found and shuffled {len(wav_files)} wav files in {input_wav_dir}")
    return wav_files

def run_tts_command(config:TTSConfig,text:str)->Optional[str]:
    try:
        command=['tts','--model_name',config.model_name,'--text',f'"{text}"','--language_idx',config.language_idx,'--out_path',config.out_path,'--device',config.device]
        if config.vocoder_name:command.extend(['--vocoder_name',config.vocoder_name])
        if config.speaker_wav:
            for wav in config.speaker_wav:command.extend(['--speaker_wav',wav])
        if config.use_cuda:command.append('--use_cuda')
        if config.source_wav:command.extend(['--source_wav',config.source_wav])
        if config.target_wav:command.extend(['--target_wav',config.target_wav])
        logging.info(f"Running TTS command: {' '.join(command)}")
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        for stdout_line in iter(process.stdout.readline,""):
            print(stdout_line,end="")
        process.stdout.close()
        return_code=process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code,command)
        return config.out_path
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred while running tts command: {str(e)}")
        return None

def move_output_file(output_path:str,output_wav_dir:str)->Optional[str]:
    try:
        if not os.path.exists(output_wav_dir):os.makedirs(output_wav_dir)
        if os.path.exists(output_path):
            new_filename=f"{uuid.uuid4()}.wav"
            new_path=os.path.join(output_wav_dir,new_filename)
            os.rename(output_path,new_path)
            logging.info(f"Moved {output_path} to {new_path}")
            return new_path
        else:
            logging.error(f"{output_path} does not exist.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while moving the file: {str(e)}")
        raise

def generate_random_text(word_file:str,word_count:int)->str:
    try:
        with open(word_file,'r') as file:words=file.readlines()
        random_words=random.sample(words,word_count)
        text=' '.join(word.strip() for word in random_words)
        logging.info(f"Generated random text: {text}")
        return text
    except Exception as e:
        logging.error(f"An error occurred while generating random text: {str(e)}")
        raise

def process_wav_file(config:TTSConfig,text:str,output_wav_dir:str)->Optional[str]:
    output_path=run_tts_command(config,text)
    if output_path:return move_output_file(output_path,output_wav_dir)
    return None

def generate_mode(args:argparse.Namespace,speaker_wav:List[str],config:TTSConfig):
    for wav_file in speaker_wav:
        logging.info(f"Processing input wav: {wav_file}")
        config.speaker_wav=[wav_file]
        for i in range(args.shots):
            result=process_wav_file(config,args.text,args.output_wav_dir)
            logging.info(f"Generated output for wav file: {wav_file}, shot: {i+1}, result: {result}")
    for i in range(args.shots):
        num_files=random.randint(args.min_merge,args.max_merge)
        selected_wav_files=random.sample(speaker_wav,min(num_files,len(speaker_wav)))
        logging.info(f"Selected {len(selected_wav_files)} speaker wav files for TTS command: {selected_wav_files}")
        config.speaker_wav=selected_wav_files
        result=process_wav_file(config,args.text,args.output_wav_dir)
        logging.info(f"Generated output for merged files, shot: {i+1}, result: {result}")

def train_mode(args:argparse.Namespace,speaker_wav:List[str],config:TTSConfig):
    for wav_file in speaker_wav:
        config.speaker_wav=[wav_file]
        for i in range(args.shots):
            text_input=generate_random_text(args.dict_file,args.word_count)
            result=process_wav_file(config,text_input,args.output_wav_dir)
            logging.info(f"Generated output for wav file: {wav_file}, shot: {i+1}, result: {result}")

def main():
    setup_logging()
    if not check_tts_command_exists():raise EnvironmentError("The 'tts' command is not available. Please install it first.")
    parser=argparse.ArgumentParser(description="TTS CLI Wrapper")
    parser.add_argument('--input_wav_dir',type=str,required=True,help='Directory of input wav files')
    parser.add_argument('--output_wav_dir',type=str,required=True,help='Directory for output wav files')
    parser.add_argument('--mode',type=str,choices=['train','generate'],required=True,help='Mode: train or generate')
    parser.add_argument('--text',type=str,help='Text to be used in generate mode')
    parser.add_argument('--dict_file',type=str,help='Dictionary file to be used in train mode')
    parser.add_argument('--shots',type=int,default=2,help='Number of shots for TTS generation')
    parser.add_argument('--word_count',type=int,default=15,help='Number of words from dictionary when training')
    parser.add_argument('--min_merge',type=int,default=2,help='Minimum number of files to merge')
    parser.add_argument('--max_merge',type=int,default=7,help='Maximum number of files to merge')
    parser.add_argument('--device',type=str,default='cpu',help='Device to use for TTS (e.g., cpu, cuda, mps)')
    parser.add_argument('--model_name',type=str,default="tts_models/multilingual/multi-dataset/xtts_v2",help='Name of the TTS model')
    parser.add_argument('--vocoder_name',type=str,help='Name of the vocoder model')
    parser.add_argument('--use_cuda',action='store_true',help='Use CUDA for TTS generation')
    parser.add_argument('--source_wav',type=str,help='Original audio file to convert in the voice of the target_wav')
    parser.add_argument('--target_wav',type=str,help='Target audio file to convert in the voice of the source_wav')
    args=parser.parse_args()
    speaker_wav=get_speaker_wav(args.input_wav_dir)
    if not speaker_wav:raise ValueError(f"No wav files found in directory {args.input_wav_dir}.")
    config=TTSConfig(model_name=args.model_name,language_idx="en",device=args.device,vocoder_name=args.vocoder_name,use_cuda=args.use_cuda,source_wav=args.source_wav,target_wav=args.target_wav)
    if args.mode=='generate':
        if not args.text:raise ValueError("Text must be provided in generate mode.")
        generate_mode(args,speaker_wav,config)
    elif args.mode=='train':
        if not args.dict_file:raise ValueError("Dictionary file must be provided in train mode.")
        train_mode(args,speaker_wav,config)

if __name__=="__main__":
    main()