function listFilesAndFolders
    folderPath='/Volumes/T9/Mac2024/CityU_Research/BV_Nov_Auditory/db3v/data_wav_8s_2/1';

    % 获取指定文件夹下的所有内容
    files = dir(folderPath);
    
    % 遍历文件夹中的每一项
    for i = 4:length(files)
        % 排除 '.' 和 '..' 这两个特殊目录
        if ~strcmp(files(i).name, '.') && ~strcmp(files(i).name, '..')
            % 构建完整路径
            currentItem = fullfile(folderPath, files(i).name);
            % 显示当前文件或文件夹的名称
            disp(currentItem);
            % 如果是文件夹，递归调用
            %feature extraction
            wav=dir(currentItem);
            for w=5:length(wav)%3
                audio=wav(w).name;
                audio_path=[wav(w).folder,'/',audio];
                [melSpecdb,gammaSpecdb]=bv_auditory_demo(audio_path);
                mdir=[currentItem(1:43),'mel/', currentItem(44:end)];
                gdir=[currentItem(1:43),'gamma/', currentItem(44:end)];
                save([mdir,'/',audio(1:end-4),'.mat'], 'melSpecdb'); 
                save([gdir,'/',audio(1:end-4),'.mat'], 'gammaSpecdb'); 
                disp([currentItem,':',num2str( (w/length(wav)) )])
            end

        end
    end
end
