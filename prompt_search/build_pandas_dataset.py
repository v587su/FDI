import os
import pickle
import tqdm
from requests_html import HTML



pairs = {}
for api_file in tqdm.tqdm(os.listdir('pandas/reference/api')):
    if api_file.endswith('.html'):
        with open(f'pandas/reference/api/{api_file}', 'r') as f:
            html = f.read()
            r = HTML(html=html)
            examples = r.find('.doctest.highlight-default.notranslate')
            api_name = api_file.split('.')[-2]                        
            query_example, code_example = None, None
            code_lines = []
            for e_div in examples:
                code_chars = e_div.find('pre', first=True).find('span')
                code_line = ''
                for i, char_span in enumerate(code_chars): 
                    if char_span.attrs and char_span.attrs['class'][0] == 'gp' and i != 0:
                        code_lines.append(code_line)
                        code_line = ''
                    elif char_span.attrs and char_span.attrs['class'][0] != 'go':
                        code_line += char_span.text 
            for code_line in code_lines:
                if api_name in code_line:
                    code_example = code_line
                    break
            query = r.find('dd p', first=True)
            if query:   
                query = query.text.strip('.')
            if query and code_example:
                pairs[query] = code_example


with open('pandas_dataset.pkl', 'wb') as f:
    pickle.dump(pairs, f)