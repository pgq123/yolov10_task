import pandas as pd
import math


if __name__ == '__main__':
    file_all = pd.read_excel(r'C:\Users\peng\Documents\yolov10\my\标准元器件深度学习库参照表.xlsx',
                             sheet_name=[i for i in range(1, 10)])
    file_name = ['PWYL', 'PWYN-C', 'PWYN-S', 'PWZF-C', 'PWZF-S']
    file_sub_name = ['CS', 'SS']
    for file1 in file_name:
        for file2 in file_sub_name:
            file_pcb = pd.read_csv(f'{file1}/{file2}/{file2}_position.csv')
            sizes = []
            classes = [[], [], [], [], [], [], [], []]
            for i in file_pcb['Shape']:
                f = 0
                for j in range(1, 10):
                    t = '元器件名称'
                    if j == 7 or j == 8:
                        t = 'Unnamed: 1'
                    for index, m in enumerate(file_all[j][t]):
                        if j == 7 and index == 36 and i == 'SUPPR-LGA-0201-P61XP31-2SM-HP165':
                            # sizes.append(file_all[j]['分辨率大小'][index])
                            classes[j].append(i)
                            f = 1
                            break
                        if (m == i or ((i == 'PADROI4516' and index == 8)
                                       or (i in ['CV+56S0MZ00', 'PADROI6332', 'X6700Y6400'] and index == 11)
                                       or (i == 'CV+56S0MZ00' and index == 12)
                                       or (i in ['CV-1094MZ01', 'PIHA5025', 'IND-PIHA052D-2SM-H2P4'] and index == 16)
                                       or (i in ['CV-2768MZ00',
                                                 'IND-IHLP404CZ-PIMA103T-MHCB10030-10P4X10P3-H3P0-COMBO'] and index == 18)
                                       or (i == 'PADROI6332' and index == 27)
                                       or (i in ['4P100_BL1000W1600', 'PADROI4P100_BL1000W1600',
                                                 '4P100_BL900W1800'] and index == 29)
                                       or (i in ['CV+4735MN04', 'PADX1700Y1500'] and index == 31)
                                       or (i in ['1608', 'CV+4740MZ06', 'PADROI2012',
                                                 'IND-P-PIXX2012-2P2X1P4-2SM-H1P2'] and index == 32)
                                       or (i in ['PADROI1005', 'PADX1100Y600'] and index == 36)
                                       or (i == 'PADX2100Y1700' and index == 37)) and j == 8):
                            # sizes.append(file_all[j]['分辨率大小'][index])
                            classes[j].append(i)
                            f = 1
                            break
                    if f:
                        break
                if not f:
                    sizes.append('NaN')
            # file_pcb['Size'] = sizes
            # file_pcb.to_csv(f'./{file1}/{file2}/{file2}_position.csv', index=False)
