function psnr=getPSNR(frameReference,frameUnderTest)
%Written by: Mahmoud Afifi ~ Assiut University, Egypt
s1=double(frameReference-frameUnderTest).^2;
    
    s = sum(sum(s1)); 
    sse = s(:,:,1)+s(:,:,2)+s(:,:,3);
    if( sse <= 1e-10) 
        psnr=0;
    else
        mse  = sse / double(size(frameReference,1)*size(frameReference,2)*size(frameReference,3));
        psnr = 10.0 * log10((255 * 255) / mse);
    end
end
