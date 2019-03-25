    function Iout = readAndPreprocessImageForGoogle(filename)  
                  
        I = imread(filename);  
        if ismatrix(I)  
            I = cat(3,I,I,I);  
        end  
           
        Iout = imresize(I, [224 224]);    
    end  