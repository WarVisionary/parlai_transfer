#!/usr/bin/python3
from parlai.scripts.display_model import DisplayModel

def main():
    model_path = '/home/warvisionary/parlai_transfer/from_pretrained/model'
    print(f"Displaying example generation from model: {model_path}")
    DisplayModel.main(
        task='empathetic_dialogues_ru', 
        model_file=model_path,

        num_examples=16,

        skip_generation=False
    )


if __name__ == "__main__":
    main()
