import pandas as pd
df = pd.read_excel('Train Data.xlsx').head(5)
with open('debug_data.txt', 'w', encoding='utf-8') as f:
    for i, row in df.iterrows():
        dept = str(row['department']).replace('\n', ' ').replace('\r', ' ').strip()
        dept = ' '.join(dept.split())
        desc = str(row['description']).replace('\n', ' ').replace('\r', ' ').strip()
        desc = ' '.join(desc.split())
        f.write(f"DEPT: {dept}\nDESC: {desc}\n\n")
