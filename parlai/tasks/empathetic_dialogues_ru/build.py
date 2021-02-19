import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import FSMTForConditionalGeneration, FSMTTokenizer

import parlai.core.build_data as build_data
from parlai.utils.io import PathManager
from parlai.tasks.empathetic_dialogues.build import build as build_en_data


class _SimpleDataset(Dataset):
    """Simple list datasetnj."""

    def __init__(self, input_utterances):
        """
        Args:
            input_utterances (List[string]): List of string utterances.
        """
        self.utterances = input_utterances

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.utterances[idx]

        return sample


def build(opt):
    version = '1.0'
    dpath = os.path.join(
        opt['datapath'],
        'empatheticdialoguesru'
    )
    
    if not build_data.built(dpath, version_string=version):
        print(f'[building data: {dpath}]')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(os.path.join(dpath, 'empatheticdialoguesru'))

        build_en_data(opt)

        mname = "facebook/wmt19-en-ru"
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        tokenizer = FSMTTokenizer.from_pretrained(mname)
        model = FSMTForConditionalGeneration.from_pretrained(mname)
        model.to(device)

        for base_datatype in ['train', 'valid', 'test']:
            en_dfpath = os.path.join(
                opt['datapath'],
                'empatheticdialogues',
                'empatheticdialogues',
                base_datatype + '.csv',
            )
            with PathManager.open(en_dfpath) as f:
                df = f.readlines()

            def _translate_utterances(utterances):
                dataset = _SimpleDataset(utterances)
                dataloader = DataLoader(dataset, batch_size=opt.get('batch_size'), shuffle=False)

                outputs = []
                for batch in dataloader:
                    tokens = tokenizer(batch, return_tensors='pt', padding=True)['input_ids']
                    outputs.append(model.generate(tokens.to(device)).to(torch.device('cpu')))
                
                translated = [
                    tokenizer.decode(output[i], skip_special_tokens=True)
                    for output in outputs
                    for i in range(output.shape[0])
                ]
                return translated

            def _translate_and_repack(utterances):
                input_utterances = [
                    utterance.replace("_comma_", ",")
                    for utterance in utterances
                ]
                translated = _translate_utterances(input_utterances)
                return [
                    utterance.replace(",", "_comma_")
                    for utterance in translated   
                ]

            dfpath = en_dfpath.replace('empatheticdialogues', 'empatheticdialoguesru')

            with PathManager.open(dfpath, mode='w') as f:
                f.write(df[0])
                turn_idx = 1
                jobs = {}
                lines = []
                lines_with_cands = {}
                for i in tqdm(range(1, len(df)), f"Translating dataset: {base_datatype}"):
                    cparts = df[i - 1].strip().split(",")
                    sparts = df[i].strip().split(",")
                    
                    # Collect turn's utterances
                    def _collect():
                        lines.append(sparts)
                        line_idx = len(lines) - 1
                        for in_line_idx in [3, 5]:
                            jobs.setdefault(sparts[in_line_idx], []).append({
                                'line_idx': line_idx,
                                'in_line_idx': in_line_idx
                            })

                        if len(sparts) == 9:
                            if sparts[8] != '':
                                in_line_idx = 8
                                for cand_idx, cand in enumerate(sparts[8].split('|')):
                                    jobs.setdefault(cand.replace("_pipe_", "|"), []).append({
                                        'line_idx': line_idx,
                                        'in_line_idx': in_line_idx,
                                        'cand_idx': cand_idx
                                    })
                                    lines_with_cands.setdefault(f"{line_idx}:{in_line_idx}", []).append(None)

                        elif len(sparts) == 8:
                            pass
                        else:
                            raise ValueError(f'Line {i:d} has the wrong number of fields!')

                    if cparts[0] == sparts[0]:
                        # Check that the turn number has incremented correctly
                        turn_idx += 1
                        assert (
                            int(cparts[1]) + 1 == int(sparts[1]) and int(sparts[1]) == turn_idx
                        )

                        _collect()
                    else:
                        # We've finished the previous episode, so translate it
                        def _translate_episode():
                            # Add indirection level to reduce memory use
                            inputs = []
                            positions = []
                            for key, value in jobs.items():
                                inputs.append(key)
                                positions.append(value)

                            if len(inputs) == 0:
                                return
                            outputs = _translate_and_repack(inputs)
                            
                            for out_idx, output in enumerate(outputs):
                                for position in positions[out_idx]:
                                    if 'cand_idx' not in position:
                                        lines[position['line_idx']][position['in_line_idx']] = output
                                    else:
                                        lines_with_cands[
                                            f"{position['line_idx']}:{position['in_line_idx']}"
                                        ][
                                            position['cand_idx']
                                        ] = output.replace("|", "_pipe_")
                            for key, value in lines_with_cands.items():
                                line_idx, pos_idx = key.split(':')
                                line_idx = int(line_idx)
                                pos_idx = int(pos_idx)
                                # Assert we found every single output that was supposed to be here
                                assert all([val is not None for val in value])
                                lines[line_idx][pos_idx] = '|'.join(value)

                            for line in lines:
                                f.write(','.join(line) + '\n')

                        _translate_episode()

                        turn_idx = 1
                        jobs = {}
                        lines = []
                        lines_with_cands = {}
                        # First utterance of any episode requires special processing
                        _collect()
                # Translate the final episode
                _translate_episode()

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
