This is a Empathetic Dialogues RU task for parlai, using original dataset for Empathetic Dialogues, translated into Russian using facebook's en-ru translator model provided in huggingface/transformers package.

Usage:

1. Merge parlai folder with parlai install folder (usually: ~/.local/lib/python3.9/site-packages/parlai)

2. Use standard parlai command to display data, translated dataset will be generated on the first run: ```python display_data.py --task empathetic_dialogues_ru```

    You can use ```--batch_size``` to change batch size for inference from default 16 that fits in "standard" 12 GiB memory of a 2080ti.

3. Data files will be generated in ~/.local/lib/python3.9/site-packages/data

4. Task can then be used regularly in parlai ecosystem

