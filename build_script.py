import base64
import glob
import gzip


def build_script(submission_name):
    script_template = open('script_template.tmpl')
    script = open('script/script_{name}.py'.format(name=submission_name), 'wt')

    file_data = {}
    for fn in glob.glob('mercari/*.py') + glob.glob('mercari/*.pyx'):
        content = open(fn).read()
        compressed = gzip.compress(content.encode('utf-8'), compresslevel=9)
        encoded = base64.b64encode(compressed).decode('utf-8')
        name = fn.split('/')[1]
        file_data[name] = encoded

    script.write(script_template.read().replace('{file_data}', str(file_data)).replace('{name}', submission_name))
    script.close()


if __name__ == '__main__':
    for submission_name in ['tf', 'mx']:
        build_script(submission_name)
