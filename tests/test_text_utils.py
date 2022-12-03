from unittest import TestCase

from utils import split_on_speaker_change, split_on_raw_utterances
from utils import timecode_re, speaker_re, punctuation_re, compute_string_similarity


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
        {SPEAKER_00}
        — Дякую, що запросив.
        
        [ 00:00:20.055 -->  00:00:35.547]
        {SPEAKER_01}
        — Це історія великого українського бізнесу, який на нашому ринку був першим, по моїм особистим враженням.
        Так і було, правильно?
        
        {SPEAKER_00}
        — Да, так.
        """

        res = split_on_raw_utterances(text)
        self.assertEquals(len(res), 3)

        # input_file = '/Users/tarassereda/data/prosody-lab/user_input_data/denys_marakin/doipislya_02_uklon/До-і-після-uklon.txt'
        input_file = '/Users/tarassereda/data/prosody-lab/user_input_data/denys_marakin/doipislya_02_uklon/uklon_en_translation_00.txt'
        with open(input_file, encoding='utf-8-sig') as fd:
            lines = fd.read()
        split_on_raw_utterances(lines)

    def test_regex(self):
        self.assertIsNotNone(timecode_re.match("[ 00:00:18.773 -->  00:00:19.904]"))
        self.assertIsNotNone(speaker_re.match("{taras_sereda}"))
        self.assertEqual('How well does this regex work', punctuation_re.sub('', 'How well,,, does this regex work?'))
        self.assertEqual('Its alright', punctuation_re.sub('', "==~~It's alright~~=="))

    def test_levenstein(self):
        str1 = "This is the story of a large Ukrainian business, which was the first in our market, according to my personal impressions. It was, right?"
        str2 = " This is the story of a large Ukrainian business, which was the first in our market according to my personal impressions. It was right?"
        self.assertEquals(compute_string_similarity(str1, str2), 1.0)
