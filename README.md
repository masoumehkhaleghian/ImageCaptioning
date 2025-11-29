# Image Captioning with Qwen 2.5-VL

This repository contains a hands-on Jupyter notebook that shows how to build an **image captioning** pipeline using the **Qwen 2.5-VL** vision-language model from Transformers.

The goal is to demonstrate, step by step, how to:

1. Load the 3B-parameter **Qwen/Qwen2.5-VL-3B-Instruct** model  
2. Send an image plus a natural-language request in **chat format**  
3. Generate a caption with `model.generate(...)`  
4. Visualize both the input image and the generated caption  

> **Runtime note**: In float-16 on GPU, the model uses roughly **6 GB VRAM**.  
> If your GPU or environment does not have enough memory, you can set  
> `device_map="cpu"` instead of `"auto"` when loading the model.

---

## Repository Structure

This repository currently contains:

- `Image-Captioning_with_Qwen 2.5-VL.ipynb`  
  A complete, runnable notebook that walks through:
  - Installing the extra Qwen VL utilities (`qwen-vl-utils`)
  - Importing all required libraries (PyTorch, Transformers, PIL, Matplotlib, etc.)
  - Loading the **Qwen2.5-VL-3B-Instruct** model and its `AutoProcessor`
  - Downloading and displaying a sample image from Unsplash
  - Building a **chat-style prompt** that mixes image and text
  - Converting the chat messages into:
    - a text prompt string (`apply_chat_template`)
    - image / video tensors (`process_vision_info`)
  - Packing everything into model-ready tensors with `processor(...)`
  - Running `model.generate(...)` to produce a caption
  - Decoding and pretty-printing the generated caption
