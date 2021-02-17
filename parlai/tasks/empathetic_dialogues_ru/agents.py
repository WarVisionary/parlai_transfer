import os

from typing import Any, List

from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher

from .build import build


DEFAULT_TRAIN_EXPERIENCER_ONLY = False
DEFAULT_REMOVE_POLITICAL_CONVOS = False


class EmpatheticDialoguesRUTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt

        self.datatype = opt['datatype']
        base_datatype = self.datatype.split(':')[0]
        self.datapath = os.path.join(
            self.opt['datapath'],
            'empatheticdialoguesru',
            'empatheticdialoguesru',
            base_datatype + '.csv',
        )
        suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        
        self.id = 'empathetic_dialogues_ru'

        self.experiencer_side_only = (
            opt.get('train_experiencer_only', DEFAULT_TRAIN_EXPERIENCER_ONLY)
            and base_datatype == 'train'
        ) or base_datatype != 'train'
        if not shared:
            print(
                f'[EmpatheticDialoguesRUTeacher] Only use experiencer side? '
                f'{self.experiencer_side_only}, datatype: {self.datatype}'
            )
        self.remove_political_convos = opt.get(
            'remove_political_convos', DEFAULT_REMOVE_POLITICAL_CONVOS
        )
        
        
        if shared:
            self.data = shared['data']
        else:
            # Actual non-boilerplate code
            build(opt)
            EmpatheticDialoguesTeacher._setup_data(self, base_datatype)

        self.num_exs = sum([len(d) for d in self.data])
        self.num_eps = len(self.data)
        self.reset()

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs

    def _select_dialogues_to_add(
        self,
        experiencer_text_dialogue: List[List[Any]],
        responder_text_dialogue: List[List[Any]],
    ) -> List[List[List[Any]]]:
        """
        Return conversation halves to add to self.data.

        Given lists corresponding to the conversation turns from both sides of the
        conversation, return only the list(s) that will be added to self.data.
        Optionally filter by side of the conversation or by whether the conversation
        contains any political language.
        """
        if self.remove_political_convos and any(
            [turn[9] for turn in experiencer_text_dialogue + responder_text_dialogue]
        ):
            return []
        else:
            selected_dialogues = []
            if len(experiencer_text_dialogue) > 0:
                selected_dialogues.append(experiencer_text_dialogue)
            if len(responder_text_dialogue) > 0 and not self.experiencer_side_only:
                selected_dialogues.append(responder_text_dialogue)
            return selected_dialogues

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        ep_i = ep[entry_idx]
        episode_done = entry_idx >= (len(ep) - 1)
        action = {
            'situation': ep_i[3],
            'emotion': ep_i[2],
            'text': ep_i[0],
            'labels': [ep_i[1]],
            'prepend_ctx': ep_i[6],
            'prepend_cand': ep_i[7],
            'deepmoji_ctx': ep_i[4],
            'deepmoji_cand': ep_i[5],
            'episode_done': episode_done,
            'label_candidates': ep_i[8],
        }
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class DefaultTeacher(EmpatheticDialoguesRUTeacher):
    pass
