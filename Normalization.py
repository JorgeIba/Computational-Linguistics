from Util import openFiles, HTMLNormalizer



all_htmls = openFiles()
normalizer = HTMLNormalizer()

# print(all_htmls[0][-350:])

normalized = []
for html in all_htmls:
    normalized += normalizer.normalize_html(html)


print(normalized[:40])

