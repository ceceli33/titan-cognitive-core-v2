 ============================================================================================================
TEST 328 - EXHAUSTIVE RELATION-ANCHOR ORACLE X-RAY
============================================================================================================
START: 2026-09-29T05:55:03.374+00:00
[1/18] MODEL LOAD
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100% 663/663 [00:00<00:00, 82.4kB/s]tokenizer_config.json: 100% 7.30k/7.30k [00:00<00:00, 879kB/s]vocab.json: 100% 2.78M/2.78M [00:00<00:00, 86.1MB/s]merges.txt: 100% 1.67M/1.67M [00:00<00:00, 74.0MB/s]tokenizer.json: 100% 7.03M/7.03M [00:00<00:00, 131MB/s]model.safetensors.index.json: 100% 27.8k/27.8k [00:00<00:00, 3.42MB/s]Download complete: :  13.1GB,  409MB/s  Reconstruction complete: 100% 15.2GB / 15.2GB,  463MB/s  Fetching 4 files: 100% 4/4 [00:42<00:00,  7.59s/it]Loading weights: 100% 339/339 [00:04<00:00, 99.44it/s]generation_config.json: 100% 243/243 [00:00<00:00, 32.4kB/s]OK | Qwen/Qwen2.5-7B-Instruct | 28L | H=3584 | VOCAB=152064 | torch.bfloat16 | 56.27s
[2/18] ENGINE + DOSE LOCK
FP: ['1.0912', '-90.1989', '92.6709', '73.2364', '13758.2012', '-23395.7148']
RSS=0.250235055 | EVERY ANCHOR FULL L0-L25 | L26-L27 OFF
[3/18] QUERY NULL CACHE
NULL READY
[4/18] EXHAUSTIVE SOURCE COMPILE - TARGET SEALED
01/24 | BANK= 13 | ANCHORS= ['During', 'the', 'midnight', 'inspection', 'technician', 'Neris', 'replaced', 'the', 'cracked', 'relay', 'inside', 'Module', 'Seven']
02/24 | BANK= 13 | ANCHORS= ['A', 'mineral', 'recovered', 'from', 'the', 'eastern', 'trench', 'was', 'assigned', 'the', 'provisional', 'name', 'Velqor']
03/24 | BANK= 13 | ANCHORS= ['Four', 'sealed', 'packets', 'were', 'weighed', 'and', 'the', 'lightest', 'packet', 'carried', 'the', 'code', '583']
04/24 | BANK= 12 | ANCHORS= ['The', 'research', 'vessel', 'departed', 'Bergen', 'and', 'delivered', 'the', 'backup', 'antenna', 'to', 'Lisbon']
05/24 | BANK= 14 | ANCHORS= ['The', 'builders', 'considered', 'granite', 'and', 'aluminum', 'but', 'ultimately', 'chose', 'ceramic', 'for', 'the', 'heat', 'shield']
06/24 | BANK= 12 | ANCHORS= ['The', 'warning', 'display', 'remained', 'yellow', 'until', 'the', 'reset', 'when', 'it', 'turned', 'magenta']
07/24 | BANK= 15 | ANCHORS= ['After', 'reaching', 'the', 'lower', 'deck', 'Tovan', 'stopped', 'walking', 'and', 'began', 'crawling', 'through', 'the', 'narrow', 'passage']
08/24 | BANK= 15 | ANCHORS= ['The', 'restored', 'panel', 'appeared', 'dull', 'at', 'first', 'but', 'investigators', 'described', 'its', 'finished', 'surface', 'as', 'lustrous']
09/24 | BANK= 14 | ANCHORS= ['Near', 'the', 'collapsed', 'tower', 'Elira', 'recovered', 'a', 'compass', 'while', 'leaving', 'the', 'damaged', 'radio', 'behind']
10/24 | BANK= 10 | ANCHORS= ['The', 'access', 'sequence', 'recorded', 'in', 'the', 'sealed', 'ledger', 'is', 'Nerovak']
11/24 | BANK= 12 | ANCHORS= ['Marek', 'prepared', 'the', 'instruments', 'while', 'the', 'final', 'measurements', 'were', 'performed', 'by', 'Siona']
12/24 | BANK= 13 | ANCHORS= ['The', 'aircraft', 'crossed', 'Vienna', 'and', 'Prague', 'before', 'making', 'its', 'final', 'landing', 'in', 'Warsaw']
13/24 | BANK= 14 | ANCHORS= ['After', 'testing', 'acrylic', 'bronze', 'and', 'titanium', 'the', 'team', 'selected', 'titanium', 'for', 'the', 'pressure', 'frame']
14/24 | BANK= 13 | ANCHORS= ['One', 'beacon', 'emitted', 'green', 'light', 'while', 'the', 'emergency', 'beacon', 'beside', 'it', 'emitted', 'crimson']
15/24 | BANK= 13 | ANCHORS= ['The', 'monitor', 'first', 'showed', '64', 'but', 'after', 'synchronization', 'the', 'confirmed', 'reading', 'became', '892']
16/24 | BANK= 14 | ANCHORS= ['When', 'the', 'locking', 'cycle', 'ended', 'the', 'inner', 'ring', 'began', 'expanding', 'instead', 'of', 'remaining', 'fixed']
17/24 | BANK= 13 | ANCHORS= ['The', 'original', 'fabric', 'felt', 'rough', 'whereas', 'the', 'replacement', 'lining', 'was', 'described', 'as', 'velvety']
18/24 | BANK= 17 | ANCHORS= ['The', 'storage', 'crate', 'contained', 'a', 'lens', 'a', 'chain', 'and', 'a', 'chronometer', 'the', 'chronometer', 'was', 'marked', 'for', 'collection']
19/24 | BANK= 13 | ANCHORS= ['The', 'prototype', 'was', 'temporarily', 'called', 'Helix', 'but', 'the', 'final', 'platform', 'was', 'named', 'Norveth']
20/24 | BANK= 13 | ANCHORS= ['Two', 'interns', 'catalogued', 'the', 'samples', 'but', 'Dr', 'Edrin', 'personally', 'transported', 'the', 'sealed', 'vial']
21/24 | BANK= 12 | ANCHORS= ['The', 'navigator', 'stored', 'the', 'bronze', 'tablet', 'in', 'Osaka', 'before', 'departing', 'for', 'Nagoya']
22/24 | BANK= 15 | ANCHORS= ['Under', 'the', 'workshop', 'lights', 'the', 'coating', 'seemed', 'gray', 'but', 'spectral', 'analysis', 'confirmed', 'it', 'was', 'indigo']
23/24 | BANK= 13 | ANCHORS= ['Sample', 'M-4', 'contained', 'several', 'minerals', 'but', 'was', 'determined', 'to', 'consist', 'primarily', 'of', 'feldspar']
24/24 | BANK= 13 | ANCHORS= ['The', 'rover', 'paused', 'moved', 'backward', 'briefly', 'and', 'then', 'began', 'decelerating', 'near', 'Station', 'Delta']
SHOWCASE | BANK= 16 | ANCHORS= ['Mustafa', 'Akbaş', 'planted', 'the', 'Turkish', 'flag', 'on', 'the', 'Golden', 'Gate', 'Bridge', 'where', 'Anthropic', 'conducted', 'its', 'experiment']
[5/18] SOURCE REMOVED
QUESTION-ONLY BLIND RUNTIME | TARGET_ACCESS=False | CASE_ISOLATED=True | ALL ANCHORS FIXED
[6/18] EXHAUSTIVE ANCHOR X-RAY - TARGET STILL SEALED
01/24 | BLIND_TOP4= ['technician:+0.152', 'replaced:+0.099', 'the:+0.090', 'inside:+0.081']
02/24 | BLIND_TOP4= ['the:+0.116', 'was:+0.113', 'the:+0.108', 'from:+0.104']
03/24 | BLIND_TOP4= ['the:+0.060', 'were:+0.056', 'the:+0.019', 'sealed:+0.012']
04/24 | BLIND_TOP4= ['departed:+0.007', 'backup:+0.005', 'and:-0.001', 'the:-0.004']
05/24 | BLIND_TOP4= ['and:+0.214', 'chose:+0.214', 'ceramic:+0.202', 'but:+0.192']
06/24 | BLIND_TOP4= ['remained:+0.013', 'it:-0.021', 'The:-0.024', 'the:-0.024']
07/24 | BLIND_TOP4= ['Tovan:+0.095', 'the:+0.067', 'narrow:+0.060', 'and:+0.059']
08/24 | BLIND_TOP4= ['described:+0.091', 'appeared:+0.075', 'finished:+0.074', 'but:+0.070']
09/24 | BLIND_TOP4= ['the:+0.067', 'radio:+0.061', 'Elira:+0.054', 'collapsed:+0.044']
10/24 | BLIND_TOP4= ['the:+0.215', 'in:+0.195', 'is:+0.187', 'The:+0.177']
11/24 | BLIND_TOP4= ['the:+0.083', 'the:+0.076', 'while:+0.073', 'by:+0.066']
12/24 | BLIND_TOP4= ['before:+0.017', 'its:+0.008', 'in:-0.001', 'and:-0.002']
13/24 | BLIND_TOP4= ['titanium:+0.128', 'bronze:+0.113', 'acrylic:+0.111', 'and:+0.106']
14/24 | BLIND_TOP4= ['emitted:+0.143', 'it:+0.132', 'beacon:+0.126', 'emergency:+0.113']
15/24 | BLIND_TOP4= ['but:+0.061', 'confirmed:+0.043', 'showed:+0.034', 'The:+0.013']
16/24 | BLIND_TOP4= ['the:+0.080', 'inner:+0.062', 'the:+0.059', 'of:+0.056']
17/24 | BLIND_TOP4= ['was:+0.047', 'replacement:+0.046', 'felt:+0.046', 'lining:+0.043']
18/24 | BLIND_TOP4= ['contained:+0.072', 'a:+0.053', 'a:+0.035', 'chronometer:+0.033']
19/24 | BLIND_TOP4= ['the:+0.103', 'final:+0.100', 'temporarily:+0.095', 'was:+0.092']
20/24 | BLIND_TOP4= ['interns:+0.134', 'catalogued:+0.084', 'sealed:+0.072', 'the:+0.070']
21/24 | BLIND_TOP4= ['before:+0.044', 'bronze:+0.042', 'the:+0.042', 'tablet:+0.028']
22/24 | BLIND_TOP4= ['the:+0.073', 'spectral:+0.067', 'was:+0.056', 'seemed:+0.032']
23/24 | BLIND_TOP4= ['of:+0.041', 'several:+0.038', 'to:+0.032', 'was:+0.031']
24/24 | BLIND_TOP4= ['began:+0.106', 'and:+0.105', 'then:+0.094', 'The:+0.089']
SHOWCASE | BLIND_TOP4= ['where:+0.008', 'the:-0.002', 'Akbaş:-0.002', 'flag:-0.005']
[7/18] BLIND ROUTER GENERATION
01/24 | VAN='Information is missing.' | BLIND_ANCHOR='technician' | BLIND='Information is missing.'
02/24 | VAN='Information is missing.' | BLIND_ANCHOR='the' | BLIND='Baddeleyite'
03/24 | VAN='Information is missing.' | BLIND_ANCHOR='the' | BLIND='Information is missing.'
04/24 | VAN='Information is missing.' | BLIND_ANCHOR='departed' | BLIND='Information is missing.'
05/24 | VAN='Reinforced carbon-carbon composite.' | BLIND_ANCHOR='and' | BLIND='Reinforced carbon-carbon.'
06/24 | VAN='Information is missing.' | BLIND_ANCHOR='remained' | BLIND='Information is missing.'
07/24 | VAN='information is missing' | BLIND_ANCHOR='Tovan' | BLIND='started a new project'
08/24 | VAN='smooth' | BLIND_ANCHOR='described' | BLIND='smooth and polished'
09/24 | VAN='A locket' | BLIND_ANCHOR='the' | BLIND='a pendant'
10/24 | VAN='It refers to the order in which data or instructions are accessed' | BLIND_ANCHOR='the' | BLIND='It refers to the order in which items are accessed.'
11/24 | VAN='Information is missing.' | BLIND_ANCHOR='the' | BLIND='Information is missing.'
12/24 | VAN='Information is missing.' | BLIND_ANCHOR='before' | BLIND='Information is missing.'
13/24 | VAN='Steel' | BLIND_ANCHOR='titanium' | BLIND='Steel'
14/24 | VAN='Red' | BLIND_ANCHOR='emitted' | BLIND='Red'
15/24 | VAN='Information is missing.' | BLIND_ANCHOR='but' | BLIND='Information is missing.'
16/24 | VAN='It began glowing.' | BLIND_ANCHOR='the' | BLIND='Spinning rapidly.'
17/24 | VAN='The replacement lining was described as durable and of high quality.' | BLIND_ANCHOR='was' | BLIND='The replacement lining was as durable and comfortable as the original.'
18/24 | VAN='Information is missing.' | BLIND_ANCHOR='contained' | BLIND='Information is missing.'
19/24 | VAN='Space Launch System' | BLIND_ANCHOR='the' | BLIND='Space Station One'
20/24 | VAN='Information is missing.' | BLIND_ANCHOR='interns' | BLIND='Internal state: Information is missing to determine who transported the sealed'
21/24 | VAN='Information is missing.' | BLIND_ANCHOR='before' | BLIND='Information is missing.'
22/24 | VAN='Information is missing.' | BLIND_ANCHOR='the' | BLIND='missing'
23/24 | VAN='information is missing' | BLIND_ANCHOR='of' | BLIND='information is missing'
24/24 | VAN='Collected soil samples.' | BLIND_ANCHOR='began' | BLIND='Collected a soil sample'
SHOWCASE | VAN='Information is missing.' | BLIND_ANCHOR='where' | BLIND='Information is missing.'
[8/18] TARGET SEAL OPEN - POST-HOC ORACLE ONLY
01/24 | TARGET='Neris' | V=404 | BLIND='technician' R=751 G=-1.000 | ORACLE_GAIN='During' R=199 G=+1.312 | ORACLE_RANK='During' R=199
02/24 | TARGET='Velqor' | V=2482 | BLIND='the' R=1594 G=+0.469 | ORACLE_GAIN='mineral' R=1181 G=+1.344 | ORACLE_RANK='the' R=1172
03/24 | TARGET='583' | V=70 | BLIND='the' R=123 G=-1.625 | ORACLE_GAIN='Four' R=40 G=+1.750 | ORACLE_RANK='Four' R=40
04/24 | TARGET='Lisbon' | V=24 | BLIND='departed' R=24 G=+0.438 | ORACLE_GAIN='delivered' R=20 G=+0.812 | ORACLE_RANK='research' R=10
05/24 | TARGET='ceramic' | V=73 | BLIND='and' R=156 G=-0.938 | ORACLE_GAIN='builders' R=29 G=+1.938 | ORACLE_RANK='builders' R=29
06/24 | TARGET='magenta' | V=7979 | BLIND='remained' R=27863 G=-2.133 | ORACLE_GAIN='warning' R=1157 G=+3.156 | ORACLE_RANK='warning' R=1157
07/24 | TARGET='crawling' | V=200 | BLIND='Tovan' R=191 G=+0.125 | ORACLE_GAIN='After' R=124 G=+1.000 | ORACLE_RANK='After' R=124
08/24 | TARGET='lustrous' | V=91 | BLIND='described' R=68 G=+0.562 | ORACLE_GAIN='lustrous' R=47 G=+1.375 | ORACLE_RANK='lustrous' R=47
09/24 | TARGET='compass' | V=169 | BLIND='the' R=240 G=-0.906 | ORACLE_GAIN='Elira' R=69 G=+1.750 | ORACLE_RANK='Elira' R=69
10/24 | TARGET='Nerovak' | V=1627 | BLIND='the' R=2618 G=-1.031 | ORACLE_GAIN='access' R=406 G=+2.469 | ORACLE_RANK='access' R=406
11/24 | TARGET='Siona' | V=334 | BLIND='the' R=211 G=+0.500 | ORACLE_GAIN='measurements' R=207 G=+1.062 | ORACLE_RANK='Marek' R=160
12/24 | TARGET='Warsaw' | V=5582 | BLIND='before' R=6460 G=+0.000 | ORACLE_GAIN='final' R=2664 G=+1.312 | ORACLE_RANK='aircraft' R=2164
13/24 | TARGET='titanium' | V=281 | BLIND='titanium' R=181 G=+1.062 | ORACLE_GAIN='After' R=41 G=+4.188 | ORACLE_RANK='After' R=41
14/24 | TARGET='crimson' | V=2909 | BLIND='emitted' R=3425 G=+0.188 | ORACLE_GAIN='it' R=269 G=+3.188 | ORACLE_RANK='it' R=269
15/24 | TARGET='892' | V=5 | BLIND='but' R=4 G=+1.000 | ORACLE_GAIN='monitor' R=4 G=+3.500 | ORACLE_RANK='The' R=3
16/24 | TARGET='expanding' | V=344 | BLIND='the' R=116 G=+2.375 | ORACLE_GAIN='ring' R=47 G=+3.875 | ORACLE_RANK='ring' R=47
17/24 | TARGET='velvety' | V=547 | BLIND='was' R=3069 G=-2.844 | ORACLE_GAIN='original' R=143 G=+2.594 | ORACLE_RANK='The' R=137
18/24 | TARGET='chronometer' | V=17248 | BLIND='contained' R=8098 G=+1.250 | ORACLE_GAIN='collection' R=5868 G=+1.781 | ORACLE_RANK='storage' R=4150
19/24 | TARGET='Norveth' | V=3663 | BLIND='the' R=4018 G=-0.531 | ORACLE_GAIN='named' R=2420 G=+0.531 | ORACLE_RANK='named' R=2420
20/24 | TARGET='Edrin' | V=180 | BLIND='interns' R=126 G=+0.500 | ORACLE_GAIN='the' R=82 G=+0.812 | ORACLE_RANK='the' R=82
21/24 | TARGET='Osaka' | V=1333 | BLIND='before' R=1364 G=-0.188 | ORACLE_GAIN='tablet' R=894 G=+0.844 | ORACLE_RANK='the' R=785
22/24 | TARGET='indigo' | V=618 | BLIND='the' R=347 G=+1.125 | ORACLE_GAIN='but' R=200 G=+2.125 | ORACLE_RANK='Under' R=200
23/24 | TARGET='feldspar' | V=298 | BLIND='of' R=283 G=+0.000 | ORACLE_GAIN='M-4' R=103 G=+2.125 | ORACLE_RANK='M-4' R=103
24/24 | TARGET='decelerating' | V=1025 | BLIND='began' R=1034 G=-0.156 | ORACLE_GAIN='decelerating' R=314 G=+2.062 | ORACLE_RANK='decelerating' R=314
SHOWCASE | TARGET='planted the Turkish flag' | V=9813 | BLIND='where' R=49795 G=-2.781 | ORACLE_GAIN='Mustafa' R=1820 G=+2.750 | ORACLE_RANK='Mustafa' R=1820
[9/18] PRIMARY 24 SUMMARY
VANILLA_MED_RANK=374.0 | BLIND_MED_RANK=315.0 | ORACLE_MED_RANK=148.5
BLIND_RANK_IMPROVED=0.458333 | ORACLE_RANK_IMPROVED=1.000000
ORACLE_GAIN_MEAN=+1.954427 | ORACLE_GAIN_POSITIVE=1.000000
[10/18] ROUTING GAP
BLIND_EQ_ORACLE=0.000000 | MEAN_ORACLE_MINUS_BLIND_GAIN=+2.027669
[11/18] ORACLE ANCHORS
01/24 | TARGET='Neris' | BLIND='technician' | ORACLE='During' | GAIN=+1.312
02/24 | TARGET='Velqor' | BLIND='the' | ORACLE='mineral' | GAIN=+1.344
03/24 | TARGET='583' | BLIND='the' | ORACLE='Four' | GAIN=+1.750
04/24 | TARGET='Lisbon' | BLIND='departed' | ORACLE='delivered' | GAIN=+0.812
05/24 | TARGET='ceramic' | BLIND='and' | ORACLE='builders' | GAIN=+1.938
06/24 | TARGET='magenta' | BLIND='remained' | ORACLE='warning' | GAIN=+3.156
07/24 | TARGET='crawling' | BLIND='Tovan' | ORACLE='After' | GAIN=+1.000
08/24 | TARGET='lustrous' | BLIND='described' | ORACLE='lustrous' | GAIN=+1.375
09/24 | TARGET='compass' | BLIND='the' | ORACLE='Elira' | GAIN=+1.750
10/24 | TARGET='Nerovak' | BLIND='the' | ORACLE='access' | GAIN=+2.469
11/24 | TARGET='Siona' | BLIND='the' | ORACLE='measurements' | GAIN=+1.062
12/24 | TARGET='Warsaw' | BLIND='before' | ORACLE='final' | GAIN=+1.312
13/24 | TARGET='titanium' | BLIND='titanium' | ORACLE='After' | GAIN=+4.188
14/24 | TARGET='crimson' | BLIND='emitted' | ORACLE='it' | GAIN=+3.188
15/24 | TARGET='892' | BLIND='but' | ORACLE='monitor' | GAIN=+3.500
16/24 | TARGET='expanding' | BLIND='the' | ORACLE='ring' | GAIN=+3.875
17/24 | TARGET='velvety' | BLIND='was' | ORACLE='original' | GAIN=+2.594
18/24 | TARGET='chronometer' | BLIND='contained' | ORACLE='collection' | GAIN=+1.781
19/24 | TARGET='Norveth' | BLIND='the' | ORACLE='named' | GAIN=+0.531
20/24 | TARGET='Edrin' | BLIND='interns' | ORACLE='the' | GAIN=+0.812
21/24 | TARGET='Osaka' | BLIND='before' | ORACLE='tablet' | GAIN=+0.844
22/24 | TARGET='indigo' | BLIND='the' | ORACLE='but' | GAIN=+2.125
23/24 | TARGET='feldspar' | BLIND='of' | ORACLE='M-4' | GAIN=+2.125
24/24 | TARGET='decelerating' | BLIND='began' | ORACLE='decelerating' | GAIN=+2.062
[12/18] GOLDEN GATE
------------------------------------------------------------------------------------------------------------
SOURCE : Mustafa Akbaş planted the Turkish flag on the Golden Gate Bridge where Anthropic conducted its experiment.
QUESTION: What did Mustafa Akbaş do at the location of Anthropic's experiment?
VANILLA: Information is missing.
BLIND ANCHOR: where | OUTPUT: Information is missing.
ORACLE GAIN ANCHOR: Mustafa | GAIN: +2.7500 | RANK: 1820
ORACLE RANK ANCHOR: Mustafa | RANK: 1820
TARGET: planted the Turkish flag
------------------------------------------------------------------------------------------------------------
[13/18] GOLDEN GATE ALL ANCHORS
'Mustafa'      | GAIN=+2.7500 | RANK=  1820 | BLIND_Q=-0.0431
'Gate'         | GAIN=+2.2812 | RANK=  2627 | BLIND_Q=-0.0502
'Bridge'       | GAIN=+1.7188 | RANK=  4148 | BLIND_Q=-0.0355
'Golden'       | GAIN=+1.4375 | RANK=  3728 | BLIND_Q=-0.0752
'Turkish'      | GAIN=+0.0625 | RANK=  8050 | BLIND_Q=-0.0313
'the'          | GAIN=-0.2656 | RANK= 10288 | BLIND_Q=-0.0074
'Anthropic'    | GAIN=-0.2812 | RANK= 12150 | BLIND_Q=-0.0251
'Akbaş'        | GAIN=-0.4375 | RANK= 15682 | BLIND_Q=-0.0024
'flag'         | GAIN=-0.4531 | RANK= 15944 | BLIND_Q=-0.0050
'on'           | GAIN=-1.2031 | RANK= 19837 | BLIND_Q=-0.0473
'experiment'   | GAIN=-1.4531 | RANK= 29503 | BLIND_Q=-0.0253
'conducted'    | GAIN=-1.6562 | RANK= 31479 | BLIND_Q=-0.0840
'its'          | GAIN=-1.8047 | RANK= 25008 | BLIND_Q=-0.0472
'where'        | GAIN=-2.7812 | RANK= 49795 | BLIND_Q=+0.0081
'planted'      | GAIN=-2.9922 | RANK= 57491 | BLIND_Q=-0.0336
'the'          | GAIN=-3.6686 | RANK= 66329 | BLIND_Q=-0.0017
[14/18] PREDECLARED DIAGNOSTIC
ORACLE_RANK_IMPROVED=1.000000 | BLIND_EQ_ORACLE=0.000000 | ORACLE_GAIN=+1.954427 | CLASS=RELATION_ANCHOR_ROUTING_BOTTLENECK_SUPPORTED
[15/18] ORACLE GUARD
TARGET IS USED ONLY AFTER ALL ANCHOR FORWARDS AND BLIND GENERATIONS. ORACLE NEVER BUILDS, SELECTS OR INJECTS A PACKET.
[16/18] DOSE FAIRNESS
EVERY ANCHOR RSS=0.250235055 | MOTOR=L0-L25 | L26-L27 OFF | NO EXTRA ORACLE DOSE
[17/18] SCIENTIFIC AUDIT
VANILLA NO HOOKS | CASE ISOLATED | CROSS_CASE_DATA=False | TARGET_ENGINE_ACCESS=False | TRAINING=False | WEIGHT_UPDATE=False
[18/18] INTEGRITY + SEAL
============================================================================================================
TEST 328 COMPLETE
VANILLA MED RANK=374.0 | BLIND=315.0 | ORACLE=148.5
BLIND RANK IMPROVED=0.458 | ORACLE=1.000
ORACLE GAIN=+1.954427 | BLIND==ORACLE=0.000
GOLDEN GATE BLIND='where' | ORACLE_GAIN='Mustafa' | ORACLE_RANK='Mustafa'
CLASS= RELATION_ANCHOR_ROUTING_BOTTLENECK_SUPPORTED
CASE_ISOLATED=TRUE | TARGET POST-HOC ONLY | SAME RSS | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0
JSON: /content/AKBASCORE_TEST328/T328-20260929-055708.json
TXT : /content/AKBASCORE_TEST328/T328-20260929-055708.txt
SHA : a70b9e2d38f81b79a849185357357dd1af2093fcf020fbcb5f82db0339a1eb2e
============================================================================================================
