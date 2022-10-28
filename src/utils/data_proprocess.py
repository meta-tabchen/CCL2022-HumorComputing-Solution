all_speakers = ['other', 'Emma', 'Others', 'lvy', 'Teddy', 'Emmett', 'Noah', 'Carlie', 'Spencer', 'Erika', 'Chalie', 'bob', 'Pj', 'Ivy', 'Dabney', 'Bob', 'Toby', 'Gas', 'Skyler', 'Jenny', 'Lauren', 'Emmet', 'others', 'othes', 'Gabe', 'Other', 'PJ', 'Beau', 'Amy', 'Charlie', '太太']

def clean_raw_sentence(text):
    if type(text) == str:
        for speaker in all_speakers:
            text = text.replace('"',"")
            text = text.replace('!?',"!")
            text = text.replace(" '",' ')
            text = text.replace(f"'{speaker}",speaker)
            text = text.replace(f"'{speaker}'",speaker)
        text = text.strip("'").strip()
    return text


