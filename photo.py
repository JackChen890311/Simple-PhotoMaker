import os
import torch
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from photomaker import PhotoMakerStableDiffusionXLPipeline

### Load base model
# base_model_type = input("Please input the base model type (sdxl/sdxl-turbo): ")
# if base_model_type == 'sdxl':
#     base_model = "stabilityai/stable-diffusion-xl-base-1.0"
# elif base_model_type == 'sdxl-turbo':
#     base_model = "stabilityai/sdxl-turbo"

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model,  # can change to any base model based on SDXL
    torch_dtype=torch.bfloat16, 
    use_safetensors=True, 
    variant="fp16"
).to(device)

photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
### Load PhotoMaker checkpoint
pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt),
    subfolder="",
    weight_name=os.path.basename(photomaker_ckpt),
    trigger_word="img"  # define the trigger word
)     

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

### Also can cooperate with other LoRA modules
# pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_model_name, adapter_name="xl_more_art-full")
# pipe.set_adapters(["photomaker", "xl_more_art-full"], adapter_weights=[1.0, 0.5])

pipe.fuse_lora()

### define the input ID images
people = input("Please input people's name: ")
while people != 'exit':
    if people not in os.listdir('./input'):
        print("Please input the right people's name!")
        people = input("Please input people's name: ")
        continue
    input_folder_name = f'./input/{people}'
    image_basename_list = os.listdir(input_folder_name)
    image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

    input_id_images = []
    for image_path in image_path_list:
        input_id_images.append(load_image(image_path))

    # Note that the trigger word `img` must follow the class word for personalization
    prompt = input("Please input the prompt (Need to have 'img' in the prompt): \n")
    while 'img' not in prompt:
        print("Please input the right prompt!")
        prompt = input("Please input the prompt (Need to have 'img' in the prompt): \n")
    # prompt = "a half-body portrait of a man img wearing the sunglasses in Iron man suit, best quality"

    negative_prompt = input("Please input the negative prompt (Don't input for default setting): \n")
    if negative_prompt == '':
        negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth, grayscale"
        # negative_prompt = "realistic, photo-realistic, bad quality, bad anatomy, worst quality, low quality, lowres, extra fingers, blur, blurry, ugly, wrong proportions, watermark, image artifacts, bad eyes, bad hands, bad arms"
    generator = torch.Generator(device=device).manual_seed(42)

    num_images = 4
    num_steps = 50
    style_strength_ratio = 20
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    images = pipe(
        prompt=prompt,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
    ).images

    os.makedirs(f'output/{people}/{prompt}', exist_ok=True)
    for i in range(num_images):
        images[i].save(f'output/{people}/{prompt}/{i}.png')
    people = input("Please input people's name: ")
