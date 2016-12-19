__kernel void rprop_plus(
    __global const float *gradients, __global float *gradients_old,
    __global float *weights, __global float *lr,
    float lr_factor_plus, float lr_factor_minus,
    float lr_limit_max, float lr_limit_min, const int length) {
    
    // Get the index of the elements to be processed
    const int i = get_global_id(0); // Index ID
    
    if(i <= length){
    
	    float elem;
	    
	    // Do the operation
	    const float g = gradients[i];
	    const float go = gradients_old[i];
	    float l = lr[i];
	    const float sign_grad = sign(g);
	    const float w = weights[i];
	    
	    elem = go * sign_grad;
	    
	    if(elem >= 0){
	        if(elem != 0){
	        	l = min(l * lr_factor_plus, lr_limit_max);
    	        lr[i] = l;
	        }
	        
	        weights[i] = w - sign_grad * l;
	        
	        gradients_old[i] = sign_grad;
	        
	    }else{
	        weights[i] = w + go * l;
	        l = max(l * lr_factor_minus, lr_limit_min);
	        lr[i] = l;
	        
	        gradients_old[i] = 0;
	    }
    }
}
