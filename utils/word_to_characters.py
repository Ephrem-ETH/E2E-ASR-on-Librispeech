lexicon_dict = {}
def lexicon_dic():
    with open("mydata/local/lm/libri_lexicon.txt","r") as f:
            for line in f:
                word = line.split()[0]
                characters = list(word)
                lexicon_dict[word] = characters
    return lexicon_dict
#print(lexicon_dict["ephrem"])
lexicon_dic()
#save the dictionary into text
trans = " "
for k in lexicon_dict:
    key =k
    transcript = lexicon_dict[k]
    #print(transcript)
    trans+=key + " " + " ".join(transcript) + "\n"
with open("amharic_lexicon.txt", "w") as wf:
    wf.write(trans)
    




        
