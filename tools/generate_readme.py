import os

base_path = '.'
url_prefix = 'Snapshot-Content-Location:'
url_404 = '#@'
link_file = 'README.md'
ref_file = 'links.md'

def get_url(file):
    with open(file) as f:
        for line in f:
            if line.startswith(url_prefix):
                for l in line.split():
                    if l.startswith('http'):
                        return l
    return url_404

def write_link(dirpath, files):
    lf = os.sep.join([dirpath, link_file])
    with open(lf, 'w') as f:
        f.write('## Links\n')
        f.writelines(files)

if __name__ == '__main__':
    rf = open(ref_file, 'w')
    rf.write('## References\n')
    for (dirpath, dirnames, filenames) in sorted(list(os.walk(base_path))):
        if dirpath == base_path:
            continue
        if dirpath.startswith(os.sep.join([base_path, '.'])):
            continue
        files = []
        rf.write(f'### {dirpath}\n')
        for filename in filenames:
            if filename.endswith('.mhtml') or filename.endswith('.mht'):
                file = os.sep.join([dirpath, filename])
                location = get_url(file)
                item = f"* [{filename}]({location})\n"
                rf.write(item)
                files.append(item)
                #print(f"* [{filename}]({location})")
        write_link(dirpath, files)
    rf.close()
