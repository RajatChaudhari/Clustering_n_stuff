from bs4 import BeautifulSoup
from pandas import DataFrame, concat
import mammoth
from os import listdir, walk
from os.path import join as join_path
from glob import glob

class pp_to_struct:
    def __init__(self):
        pass

    def tbl1_to_df(self,html):
        tbl1_dict = dict()
        for row in html.find_all('table')[1].find_all('tr'):
                cells = row.find_all('td')
                key = ' '.join([p.text for p in cells[0].find_all('p')])
                del cells[0]
                value = ' '.join([p.text for p in [cell.find_all('p') for cell in cells][0]]).replace('\n', '').replace('\t', '')
                tbl1_dict[key] = value
        return DataFrame(tbl1_dict, index=[0])

    def tbl2_to_df(self,html):
        tbl_2_dict = dict()
        rows = html.find_all('table')[2].find_all('tr')
        responsabilities = []
        for index in range(0, len(rows)):
                row = rows[index]
                cells = row.find_all('td')
                if len(cells) > 1:
                    if index == 0:
                        key = cells[0].text
                        value = cells[1].text
                        tbl_2_dict[key] = value
                    elif index != 1:
                        value = cells[0].text
                        responsabilities.append(value)
        responsabilities_str = '$$$%%%&&&'.join(responsabilities)
        tbl_2_dict['responsabilities'] = responsabilities_str
        return DataFrame(tbl_2_dict, index = [0])


    def tbl9_to_df(self,html):
        tbl_9_dict = dict()
        rows = html.find_all('table')[9].find_all('tr')
        competency_group = rows[1].text.replace('Select Competency Group: ', '')
        competencies = []
        for index in range(2, len(rows)):
            row = rows[index]
            cells = row.find_all('li')
            for cell in cells:
                competencies.append(cell.text)
        competencies_str = '$$$%%%&&&'.join(competencies)
        tbl_9_dict['competencies'] = competencies_str
        tbl_9_dict['competency_group'] = competency_group
        #return DataFrame.from_dict(tbl_9_dict, orient='index', columns = ['competencies', 'competency_group'])
        return DataFrame(tbl_9_dict, index=[0])

    def tbls_reqd_assets_df(self,html, table_index, type_of_qualifications):
        tbl_dict = dict()
        header = html.find_all('table')[table_index].find('th').text
        qualifications = []
        for row in html.find_all('table')[table_index].find_all('tr'): 
            cells = row.find_all('td')
            if cells and len(cells) == 3:
                key = cells[0].text
                if key:
                    qualifications.append(key)
        tbl_dict[type_of_qualifications] = '$$$%%%&&&'.join(qualifications)
        return DataFrame(tbl_dict, index=[0])

    def only_reqd_df(self,html, table_index, type_of_qualifications):
        tbl_dict = dict()
        header = html.find_all('table')[table_index].find('th').text
        qualifications = []
        for row in html.find_all('table')[table_index].find_all('tr'): 
            cells = row.find_all('td')
            if cells and len(cells) == 3:
                key = cells[0].text
                reqd = cells[1].text
                if reqd and key:
                    qualifications.append(key)
        tbl_dict[type_of_qualifications] = '$$$%%%&&&'.join(qualifications)
        return DataFrame(tbl_dict, index=[0])

    def only_asset_df(self,html, table_index, type_of_qualifications):
        tbl_dict = dict()
        header = html.find_all('table')[table_index].find('th').text
        qualifications = []
        for row in html.find_all('table')[table_index].find_all('tr'): 
            cells = row.find_all('td')
            if cells and len(cells) == 3:
                key = cells[0].text
                asset = cells[2].text
                if asset and key:
                    qualifications.append(key)
        tbl_dict[type_of_qualifications] = '$$$%%%&&&'.join(qualifications)
        return DataFrame(tbl_dict, index=[0])

    def tbl_opts_df(self,html, table_index):
        tbl_dict = dict()
        for row in html.find_all('table')[table_index].find_all('tr'):
            cells = row.find_all('td')
            if cells:
                i = 0
                while(i < len(cells)):
                    try:
                        key = cells[i].text
                        i+=1
                        if key:
                            value = cells[i].text
                            value = value if value != '<Select>' else None
                            tbl_dict[key] = value
                        i+=1
                    except:
                        pass
        return DataFrame(tbl_dict, index=[0])

    def html_to_df(self,html_str):
        html = BeautifulSoup(html_str, 'html.parser')
        if len(html.find_all('table')) == 17:
            tbl1 = self.tbl1_to_df(html)
            tbl2 = self.tbl2_to_df(html)
            tbl4 = self.tbls_reqd_assets_df(html, 4, 'CA')
            tbl4_reqd = self.only_reqd_df(html, 4, 'CA_REQD')
            tbl4_asset = self.only_asset_df(html, 4, 'CA_ASSET')
            tbl5 = self.tbls_reqd_assets_df(html, 5, 'CP')
            tbl5_reqd = self.only_reqd_df(html, 5, 'CP_REQD')
            tbl5_asset = self.only_asset_df(html, 5, 'CP_ASSET')
            tbl6 = self.tbls_reqd_assets_df(html, 6, 'ER')
            tbl6_reqd = self.only_reqd_df(html, 6, 'ER_REQD')
            tbl6_asset = self.only_asset_df(html, 6, 'ER_ASSET')
            tbl7 = self.tbls_reqd_assets_df(html, 7, 'EI')
            tbl7_reqd = self.only_reqd_df(html, 7, 'EI_REQD')
            tbl7_asset = self.only_asset_df(html, 7, 'EI_ASSET')
            tbl8 = self.tbls_reqd_assets_df(html, 8, 'TS')
            tbl8_reqd = self.only_reqd_df(html, 8, 'TS_REQD')
            tbl8_asset = self.only_asset_df(html, 8, 'TS_ASSET')
            tbl9 =  self.tbl9_to_df(html)
            tbl12 =  self.tbl_opts_df(html, 12)
            tbl13 =  self.tbl_opts_df(html, 13)
            tbl14 =  self.tbl_opts_df(html, 14)
            tbl15 =  self.tbl_opts_df(html, 15)
            tbl16 =  self.tbl_opts_df(html, 16)
            all_tbls = [tbl1, tbl2, tbl4, tbl4_reqd, tbl4_asset, tbl5, tbl5_reqd, tbl5_asset, tbl6, tbl6_reqd, tbl6_asset,
                        tbl7, tbl7_reqd, tbl7_asset, tbl8, tbl8_reqd, tbl8_asset, tbl9, tbl12, tbl13, tbl14, tbl15, tbl16]
            return concat(all_tbls, axis = 1, join='inner')
        else:
            return DataFrame()





