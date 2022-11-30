from unittest import TestCase

from utils import split_on_speaker_change, convert_text_to_segments
from utils import timecode_re, speaker_re


class Test(TestCase):
    def test_split_on_speaker_change(self):
        text = """
        {taras_sereda}
        November 1, nice sunny day in Kyiv - capital of brave and resilient people.
        {joe_rogan}
        Talking, here again
        {taras_sereda}
        Another view on that problem
        {joe_rogan}
        see you in Cali!
        """
        res = split_on_speaker_change(text)
        res = list(res)
        self.assertEquals(res[0][0].strip(),
                          "taras_sereda")
        self.assertEquals(res[0][1].strip(),
                          "November 1, nice sunny day in Kyiv - capital of brave and resilient people.")
        self.assertEquals(res[2][0].strip(),
                          "taras_sereda")
        self.assertEquals(res[2][1].strip(),
                          "Another view on that problem")
        self.assertEquals(res[3][0].strip(),
                          "joe_rogan")
        self.assertEquals(res[3][1].strip(),
                          "see you in Cali!")

    def test_convert_text_to_segments(self):
        text = """
        [ 00:00:18.773 -->  00:00:19.904]
        [SPEAKER_00]
        — Дякую, що запросив.
        
        [ 00:00:20.055 -->  00:00:35.547]
        [SPEAKER_01]
        — Це історія великого українського бізнесу, який на нашому ринку був першим, по моїм особистим враженням.
        Так і було, правильно?
        
        [SPEAKER_00]
        — Да, так.
        """

        res = convert_text_to_segments(text)
        self.assertEquals(len(res), 3)

        # input_file = '/Users/tarassereda/data/prosody-lab/user_input_data/denys_marakin/doipislya_02_uklon/До-і-після-uklon.txt'
        # with open(input_file, encoding='utf-8-sig') as fd:
        #     lines = fd.read()
        # convert_text_to_segments(lines)

    def test_regex(self):
        self.assertIsNotNone(timecode_re.match("[ 00:00:18.773 -->  00:00:19.904]"))
        self.assertIsNotNone(speaker_re.match("{taras_sereda}"))
