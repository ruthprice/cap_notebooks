suites = [
    'u-cr700',
   'u-cu657',
   'u-ct323',
   'u-cw067',
    'u-cw894',
   'u-dc034',
   'u-dd355'
]

configs = [
    '',
   '_ra3_ukca_casim',
   '_fixed_drop_150',
   '_RAL3p1_ukca',
    '_RAL3p1_ukca',
   '_RAL3p1_ukca',
   '_RAL3p1_ukca',
]

stashcode = [
    'm01s00i004',
    'm01s00i025',
    'm01s00i408',
    'm01s00i075',
    'm01s00i254',
    'm01s00i267',
    'm01s01i235',
    'm01s02i207',
    'm01s34i101',
    'm01s34i102',
    'm01s34i103',
    'm01s34i104',
    'm01s34i105',
    'm01s34i106',
    'm01s34i107',
    'm01s34i109',
    'm01s34i110',
    'm01s34i111',
    'm01s34i113',
    'm01s34i114',
    'm01s34i115',
    'm01s34i116',
    'm01s34i117',
    'm01s34i118',
    'm01s34i119',
    'm01s34i120',
    'm01s34i121',
    'm01s34i126',
    'm01s38i401',
    'm01s38i402',
    'm01s38i403',
    'm01s38i404',
    'm01s38i405',
]

suites_fn = 'suites.csv'
stash_fn = 'stashcodes.csv'
suites_output = ''
for s,suite in enumerate(suites):
    suites_output += f'{suite}, {configs[s]}\n'
stash_output = ''
for stash in stashcode:
    stash_output += f'{stash}\n'

with open(suites_fn, 'w') as file_writer:
    file_writer.write(suites_output)
file_writer.close()

with open(stash_fn, 'w') as file_writer:
    file_writer.write(stash_output)
file_writer.close()
