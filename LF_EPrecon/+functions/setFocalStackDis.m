function fsDis = setFocalStackDis(workRange, nF, option, f,varargin)
    switch option
        case 'linear'
            %f = 50;   
            W12 = [max(workRange) min(workRange)];
            B12 = 1./(1/f - 1./W12);            
            fsDis = linspace(B12(1), B12(2), nF);            
            %fsDis = linspace(min(workRange), max(workRange), nF);            
        case 'even Fourier slice'
            %W12 = [2700 1800];
            %f = 50;
            f=100;
            N = nF;
            
            W12 = [max(workRange) min(workRange)];
            B12 = 1./(1/f - 1./W12);
            %B12 is the image space location where W12 are sharply focused

            W1 = W12(1);
            W2 = W12(2);
            
            syms b
            eqn = ( pi + atan(1/(b/f-b/W2-1)) - atan(1/(b/f-b/W1-1)) )/(2*N) == pi/2 + atan(1/(b/f-b/W2-1));
            %beta = double( vpasolve(eqn, b, 51.38) );

            beta = double( vpasolve(eqn, b, B12) );
            %here beta seems to be just F, the distance between the lens
            %and the last focal stack, the above equation solves to get the
            %beta, i.e., the last focal stack position (note F is also the 2nd reference plane of the light field )
            thetaW1 = atan(1/(beta/f-beta/W1-1));
            thetaW2 = pi + atan(1/(beta/f-beta/W2-1));            
            
            dtheta = (thetaW2-thetaW1)/N;
            rad2deg(dtheta)

            d = 1:N;
            thetad = thetaW1+dtheta/2 + (d-1)*dtheta;%all sampled fourier space angles

            fsDis = beta*tan(thetad)./(1 + tan(thetad));    
       
        case 'linear_disp'
            %setFocalStackdis based on the linear disparity interval, using
            %d=uF(1/z'-1/z), where F is the refDis, z is the original
            %focusing distance, z' is the refocusing distance, u is the
            %baseline, d is the disparity   
             % in this case, the input argument should have form: fsDis = setFocalStackDis([], nF, 'linear_disp', f,dx,du,focusingDis,refDis,dispRange)
             %where dispRange should has form [disp_min, disp_max]
             assert (isempty(workRange),'WorkRange should be empty in this case');
             dx=varargin{1};
             du=varargin{2};
             focusingDis=varargin{3};
             refDis=varargin{4};
             dispRange=varargin{5};
             
             z_prime_minmax=1./(dx*dispRange/(du*refDis)+1/focusingDis);
             z_primeRange= linspace(z_prime_minmax(1),z_prime_minmax(2), nF);     
             fsDis=1./(1/f-1./z_primeRange);
             
         case 'disp_list'
            %setFocalStackdis based on the  disparity list, using
            %d=uF(1/z'-1/z), where F is the refDis, z is the original
            %focusing distance, z' is the refocusing distance, u is the
            %baseline, d is the disparity   
             % in this case, the input argument should have form: fsDis = setFocalStackDis([], [], 'disp_list', f,dx,du,focusingDis,refDis,dispList)
             %where dispList should has form [d1,d2,... d_nF]
             assert (isempty(workRange),'WorkRange should be empty in this case');
             assert (isempty(nF),'nF should be empty in this case');
             dx=varargin{1};
             du=varargin{2};
             focusingDis=varargin{3};
             refDis=varargin{4};
             dispList=varargin{5};
             
             z_prime_list=1./(dx*dispList/(du*refDis)+1/focusingDis);
             fsDis=1./(1/f-1./z_prime_list);
             
            
%{            
            W1 = max(workRange);
            W2 = min(workRange);
            N = nF;
            f = 50;

            syms b
            eqn = ( pi + atan(1/(b/f-b/W2-1)) - atan(1/(b/f-b/W1-1)) )/(2*N) == pi/2 + atan(1/(b/f-b/W2-1));
            beta = double( vpasolve(eqn, b) );

            %gamma12 = 1./(1/f - 1./[W1 W2]);

            thetaW1 = atan(1/(beta/f-beta/W1-1));
            rad2deg(thetaW1)
            thetaW2 = pi + atan(1/(beta/f-beta/W2-1));
            rad2deg(thetaW2)
            dtheta = (thetaW2-thetaW1)/N;
            rad2deg(dtheta)

            d = 1:N;
            thetad = thetaW1+dtheta/2 + (d-1)*dtheta;
            rad2deg(thetad)

            beta
            dis = beta*tan(thetad)./(1 + tan(thetad))
            ddis = dis(2:end) - dis(1:end-1);

            % Baseline scene
            thetaB = [thetaW1+2*dtheta thetaW1+3*dtheta];
            disB = beta*tan(thetaB)./(1 + tan(thetaB));
            sceneB = 1./(1/f - 1./disB) 
            fsDis = disB
%}            
        otherwise
            error('Invalid/Null option name.');
    end    
end