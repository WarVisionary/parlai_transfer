#!/usr/bin/python3
from parlai.scripts.interactive import Interactive

def main():
    model_path = '/home/warvisionary/parlai_transfer/from_pretrained/model'
    print(f"Interacting with model at: {model_path}")
    Interactive.main(
        task='empathetic_dialogues_ru', 
        model_file=model_path
    )


if __name__ == "__main__":
    main()
