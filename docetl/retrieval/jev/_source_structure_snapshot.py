"""Query-independent numbering and title recovery on immutable plain text.

No tasks, questions, annotations, document names or provider/model calls.
Candidates are structural heuristics, not human-verified semantic headings.
"""
import re

WORDS=['ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT','NINE','TEN','ELEVEN','TWELVE','THIRTEEN','FOURTEEN','FIFTEEN','SIXTEEN','SEVENTEEN','EIGHTEEN','NINETEEN','TWENTY']
ROMAN=r'[IVXLCDM]+'
BODY=re.compile(r'\b(?:shall|means|agree|made|entered|will|is|are|may)\b',re.I)


def number(value):
    v=value.upper().rstrip('.')
    if v.isdigit():return int(v)
    if v in WORDS:return WORDS.index(v)+1
    if re.fullmatch(ROMAN,v):
        values={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000};out=previous=0
        for c in reversed(v):
            n=values[c];out+=-n if n<previous else n;previous=max(previous,n)
        return out
    return None


def toc_ranges(text):
    out=[]
    paragraphs=list(re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)',text))
    for i,p in enumerate(paragraphs):
        if not re.search(r'\bTABLE\s+OF\s+CONTENTS\b',p.group(),re.I):continue
        end=p.end()
        for q in paragraphs[i+1:]:
            value=q.group().strip();words=re.findall(r'\b[A-Za-z]+\b',value)
            lower=sum(w[0].islower() for w in words)
            if len(words)>=15 and lower/len(words)>.35 and BODY.search(value) and not value.startswith('Source:'):break
            end=q.end()
        out.append((p.start(),end))
    return out


def title_prefix(rest):
    """Keep only a short source title; numbers remain valid for untitled clauses."""
    definition=re.match(r'(["“][^"”\n]{1,80}["”])\s+(?:means|shall mean|is defined)',rest,re.I)
    if definition:return definition.group(1)
    stop=re.search(r'[.:]\s+(?=[A-Z"“(])',rest)
    prefix=rest[:stop.start()+1] if stop else rest.split('\n',1)[0].strip()
    if not prefix or len(prefix)>120 or len(prefix.split())>16 or BODY.search(prefix):return ''
    return prefix


def parse(text):
    if not isinstance(text,str) or not text:raise ValueError('Nonempty original source required')
    toc=toc_ranges(text);candidates=[];excluded=[]
    boundary=r'(?:^[ \t]*|[ \t]{2,})'
    article=re.compile(boundary+r'(?P<label>ARTICLE\s+(?P<num>\d{1,3}|'+ROMAN+'|'+ '|'.join(sorted(WORDS,key=len,reverse=True))+r')\b\.?)(?P<rest>[^\n]*)',re.M|re.I)
    numeric=re.compile(boundary+r'(?P<label>(?:SECTION[ \t]+)?(?P<num>\d{1,3}(?:\.\d{1,3})*)[.)]?)[ \t]+(?P<rest>[A-Za-z"“][^\n]*)',re.M|re.I)
    for kind,pattern in [('article',article),('clause',numeric)]:
        # Look ahead at every candidate: a flattened line may hold many clauses.
        search=re.compile(r'(?='+pattern.pattern+r')',pattern.flags)
        for m in search.finditer(text):
            start=m.start('label');label=m.group('label');rest=m.group('rest').strip()
            if any(a<=start<b for a,b in toc):excluded.append({'start':start,'label':label,'reason':'table_of_contents_region'});continue
            preceding=text[text.rfind('\n',0,start)+1:start]
            inline=bool(preceding.strip())
            if inline and (rest[:1].islower() or re.search(r'\b(?:see|section|sections|under|article|in|to)\s*$',preceding,re.I)):
                excluded.append({'start':start,'label':label,'reason':'inline_reference_or_lowercase_continuation'});continue
            # Numeric years/page labels and common date-like lines are not clauses.
            path=tuple(int(x) for x in m.group('num').split('.')) if kind=='clause' else (number(m.group('num')),)
            if path[0] is None or path[0]<=0 or path[0]>999:continue
            if kind=='clause' and re.match(r'(?:of\s+\d|pages?\b|January\b|February\b|March\b|April\b|May\b|June\b|July\b|August\b|September\b|October\b|November\b|December\b)',rest,re.I):continue
            title=title_prefix(rest)
            if kind=='article':
                # Title can precede an inline first clause on the same line.
                first=re.search(r'\s{2,}\d+(?:\.\d+)+\s',rest)
                if first:title=rest[:first.start()].strip()
                elif not title and rest and len(rest)<120 and not BODY.search(rest):title=rest
                elif not rest:
                    following=re.match(r'[ \t]*\n(?:[ \t]*\n)*[ \t]*([^\n]+)',text[start+len(label):])
                    if following:
                        value=following.group(1).strip()
                        if 3<=len(value)<=100 and value.isupper() and not BODY.search(value):title=value
            title_end=start+len(label)
            if title:
                found=text.find(title,title_end,min(len(text),title_end+300))
                if found>=0:title_end=found+len(title)
            candidates.append({'start':start,'title_end':title_end,'title':text[start:title_end].strip(),
                               'kind':kind,'number_path':list(path),'inline':inline})
    # A clause with a clear prefix is useful even when it shares a long body line.
    unique={}
    for c in sorted(candidates,key=lambda c:(c['start'],c['kind']!='article')):unique.setdefault(c['start'],c)
    candidates=list(unique.values())
    root={'id':'TEXT_ROOT','start':0,'end':len(text),'title':'Contract','children':[],'kind':'root'}
    stack=[root]
    for i,c in enumerate(candidates):
        if c['kind']=='article':
            parent=root;stack=[root]
        else:
            path=c['number_path']
            # Numbers nest by actual prefix, not by decimal depth alone.
            while len(stack)>1:
                prev=stack[-1]
                if prev['kind']=='article':
                    if path[0]==prev['number_path'][0]:break
                    stack.pop();continue
                before=prev['number_path']
                if len(before)<len(path) and path[:len(before)]==before:break
                stack.pop()
            parent=stack[-1]
        node={**c,'id':f'TEXT_H{i}_{c["start"]}','end':len(text),'children':[]}
        parent['children'].append(node);stack.append(node)
    def close(node):
        for i,c in enumerate(node['children']):
            c['end']=node['children'][i+1]['start'] if i+1<len(node['children']) else node['end']
            close(c)
    close(root)
    excluded=list({(e['start'],e['reason']):e for e in excluded}.values())
    return {'semantic':root,'candidates':candidates,'excluded_candidates':excluded,'toc_ranges':toc,
            'source_only':True,'classification':'deterministic structural heuristics; no model votes'}
