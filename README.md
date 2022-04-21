# MTL-ViT: A new multi-task learning framework using Vision Transformers
(*Note: This is an ongoing project, hence the full code and strategy is not yet open-sourced by the author.)

We presnet a new multi-task learning strategy using Vision transformers (ViTs). Our approach is based on exploiting the class-token and self-attention mechanism of Vision Transformers in order to train multiple tasks through a single ViT, more efficiently and with limited computational budget.


![alt text](https://github.com/hananshafi/MTL-ViT/blob/main/assets/network.jpg)

### Total Loss of the Multi-task system: <img src="https://render.githubusercontent.com/render/math?math=L_{total}=L_{1}%2BL_{1}%2BL_{3}%2B . . . %2B L_{n} "> 
