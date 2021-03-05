from scrap import wiki_page, wiki_scrape
from make_graph import draw_kg
from get_pairs import get_entity_pairs

pfizer_data = wiki_page('Pfizer')
pairs = get_entity_pairs(pfizer_data['text'][0])
draw_kg(pairs)