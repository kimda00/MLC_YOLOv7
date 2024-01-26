import os

# 폴더들의 경로
folder_paths = [
    './croped/black', './croped/green', './croped/green_arrow','./croped/green_right', 
    './croped/greenyellow', './croped/red', './croped/red_arrow', './croped/red_yellow','./croped/yellow']
# folder_paths = ['black_light_testset']
# 각 폴더를 순회하며 이미지 파일들의 이름 변경
for folder_path in folder_paths:
    # 해당 폴더 내 모든 파일 리스트 가져오기
    file_list = os.listdir(folder_path)
    
    # 이미지 파일들만 골라내어 처리
    image_files = [f for f in file_list if f.lower().endswith('.jpg')]
    
    # 이미지 파일들의 이름 변경
    for idx, image_file in enumerate(image_files):
        new_name = f'{os.path.basename(folder_path)}_{idx+1}.jpg'
        old_path = os.path.join(folder_path, image_file)
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        print(f'Renamed: {old_path} -> {new_path}')