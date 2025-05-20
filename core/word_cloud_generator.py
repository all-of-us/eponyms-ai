import matplotlib.pyplot as plt
from pandas import read_csv
from wordcloud import WordCloud
from collections import Counter

from resources.constants import data_path


def main(frequencies):
    wc = WordCloud(width=3200, height=1600, background_color=None, contour_color=None, mode='RGBA', colormap='ocean',
                   collocations=False)
    wc.generate_from_frequencies(frequencies=frequencies)
    plt.figure(figsize=(20, 10), facecolor=None)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(data_path / 'final_eponymous_concepts.png', transparent=True)


if __name__ == '__main__':
    freq_path = data_path / 'final_eponymous_concepts.csv'

    from nltk.corpus import stopwords

    df = read_csv(freq_path, names=['concept_name', 'concept_id'], header=0)
    df['concept_name'] = df['concept_name']
    rows_string = df['concept_name'].str.cat(sep=" ").replace("(", " ").replace("[", " ").replace("]", " ").replace(")",
                                                                                                                    " ").replace(
        ":", " ").replace("/", " ").replace("\\", " ").replace("&", " ").replace(",", "").lower()
    rows_string = ' '.join([word for word in rows_string.split() if word not in stopwords.words("english")])
    freq_df = df.from_dict(Counter(rows_string.split()), orient='index', columns=['count'])
    # freq_df = freq_df['ne'].dropna()
    freq_df['ne'] = freq_df.index
    # df['ne'] = df['ne'].str.title()
    freqs = dict(zip(freq_df['ne'], freq_df['count']))
    main(freqs)
