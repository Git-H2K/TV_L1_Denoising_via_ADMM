function Result = TV_L1_ADMM(input, lambda, cost_threshold, max_Iter)
Img_ori = im2double(input);
[height,width,ch] = size(input);

denom_tmp = (abs(psf2otf([1, -1],[height,width])).^2 + abs(psf2otf([1; -1],[height,width])).^2);
if ch~=1
    denom_tmp = repmat(denom_tmp, [1 1 ch]);
end

% Initialize Vraiables
Diff_R_I = zeros(size(Img_ori));   % Result - Input
grad_x = zeros(size(Img_ori));   
grad_y = zeros(size(Img_ori));   

aux_Diff_R_I = zeros(size(Img_ori));
aux_grad_x = zeros(size(Img_ori));
aux_grad_y = zeros(size(Img_ori));

Diff = 100000; % Initialize Diff
Cost_prev = 10^5; % Initialize Cost
alpha = 0.02;
beta = 0.02;
Iter = 0;

% ADMM
while Diff > cost_threshold || Iter < max_Iter
    grad_x_tmp = grad_x + aux_grad_x/alpha;
    grad_y_tmp = grad_y + aux_grad_y/alpha;
    
    numer_alpha = fft2(Diff_R_I+ aux_Diff_R_I/beta) + fft2(Img_ori);
    numer_beta = [grad_x_tmp(:,end,:) - grad_x_tmp(:, 1,:), -diff(grad_x_tmp,1,2)];
    numer_beta = numer_beta + [grad_y_tmp(end,:,:) - grad_y_tmp(1, :,:); -diff(grad_y_tmp,1,1)]; 
    
    denomin = 1 + alpha/beta*denom_tmp;
    numer = numer_alpha+alpha/beta*fft2(numer_beta);
    
    Result = real(ifft2(numer./denomin));    
    Result_x = [diff(Result,1,2), Result(:,1,:) - Result(:,end,:)]; 
    Result_y = [diff(Result,1,1); Result(1,:,:) - Result(end,:,:)]; 
    
    grad_x = Result_x - aux_grad_x/alpha;
    grad_y = Result_y - aux_grad_y/alpha;
    
    Mag_grad_x = abs(grad_x);
    Mag_grad_y = abs(grad_y);
    if ch~=1
        Mag_grad_x = repmat(sum(Mag_grad_x,3), [1,1,ch]);
        Mag_grad_y = repmat(sum(Mag_grad_y,3), [1,1,ch]);
    end
    
    grad_x = max(Mag_grad_x-lambda/alpha,0).*(grad_x./Mag_grad_x);
    grad_y = max(Mag_grad_y-lambda/alpha,0).*(grad_y./Mag_grad_y);
    grad_x(Mag_grad_x == 0) = 0;
    grad_y(Mag_grad_y == 0) = 0;
    
    Diff_R_I = Result-Img_ori-aux_Diff_R_I/beta;
    
    Mag_Diff_R_I = abs(Diff_R_I);    
    if ch~=1
        Mag_Diff_R_I = repmat(sum(Mag_Diff_R_I,3), [1,1,ch]);
    end
    
    Diff_R_I=max(Mag_Diff_R_I-1/beta,0).*(Diff_R_I./Mag_Diff_R_I);
    Diff_R_I(Mag_Diff_R_I == 0) = 0;
        
    aux_Diff_R_I = aux_Diff_R_I + beta * (Diff_R_I - (Result - Img_ori ));
    aux_grad_x = aux_grad_x + alpha * (grad_x - (Result_x ));
    aux_grad_y = aux_grad_y + alpha * (grad_y - (Result_y));
    
    alpha = alpha+0.07;
    beta = beta+0.07;
    
    Result_x = [diff(Result,1,2), Result(:,1,:) - Result(:,end,:)];
    Result_y = [diff(Result,1,1); Result(1,:,:) - Result(end,:,:)];
    
    Cost_cur = sum(abs(Result(:) - Img_ori(:))) + lambda*sum(abs(Result_x(:)) + abs(Result_y(:)));
    Diff = abs(Cost_cur - Cost_prev);
    Cost_prev = Cost_cur;    
    Iter = Iter + 1;
    
    fprintf('Diff : %f, Cost : %f \n', Diff, Cost_cur);
end
end