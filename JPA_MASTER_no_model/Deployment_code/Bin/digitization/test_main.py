import family_classifier,jp_struct,pp_struct
import mammoth


convertor = pp_struct.pp_to_struct()
clf = family_classifier.family_classify()



path = 'Manager, Human Resources Business Partners - Human Resources Business Partners - Human Resources.DOCX'

file = open(path, 'rb') 

fils = mammoth.convert_to_html(file).value

family = convertor.html_to_df(fils)

print(clf.clf(family),'\n',family)
