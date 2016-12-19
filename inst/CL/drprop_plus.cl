__kernel void rprop_plus(
    __global const double *gradients, __global double *gradients_old,
    __global double *weights, __global double *lr,
    double lr_factor_plus, double lr_factor_minus,
    double lr_limit_max, double lr_limit_min, const int length) {
    
    // Get the index of the elements to be processed
    const int i = get_global_id(0); // Index ID
    
    if(i <= length){
    
        //printf("index: %d\n", i);
    
	    double elem;
	    
	    // Do the operation
	    const double g = gradients[i];
	    const double go = gradients_old[i];
	    double l = lr[i];
	    const double sign_grad = sign(g);
	    const double w = weights[i];
	    
	    //if(go != 0){
		    elem = go * sign_grad;
	    //}else{
	    //	elem = 0;
	    //}
	    
	    if(i == 31){
	    	//printf("elem: %f\n", elem);
	    	//printf("g: %f\n", g);
	    	//printf("go: %f\n", go);
	    	//printf("lr: %f\n", l);
	    	//printf("w: %f\n", w);
	    	//printf("sg: %f\n", sign_grad);
	    	//printf("minus: %f\n", lr_factor_minus);
	    	//printf("min: %f\n", lr_limit_min);
	    	//printf("plus: %f\n", lr_factor_plus);
	    	//printf("max: %f\n", lr_limit_max);
	    }
	    
	    if(elem >= 0){
	        if(elem != 0){
	        		l = min(l * lr_factor_plus, lr_limit_max);
    	        lr[i] = l;
    	        //printf("positive lr update: %f\n", l * lr_factor_plus);
	        }
	        
	        //printf("lr for update: %f\n", l);
	        weights[i] = w - sign_grad * l;
	        
	        if(i == 31){
	          //printf("lr: %f\n", lr[i]);
	          //printf("positive calculated: %f\n", w - sign_grad * l);
	          //printf("updated weight: %f\n", weights[i]);
	        }
	        
	        gradients_old[i] = sign_grad;
	        
	    }else{
	        weights[i] = w + go * l;
	        l = max(l * lr_factor_minus, lr_limit_min);
	        lr[i] = l;
	        
	        
	        if(i == 31){
	          //printf("negative elem: %f\n", elem);
	          //printf("negative calculated: %f\n", w - sign_grad * l);
	          //printf("updated weight: %f\n", weights[i]);
	        }
	        
	        gradients_old[i] = 0;
	    }
    }
}
