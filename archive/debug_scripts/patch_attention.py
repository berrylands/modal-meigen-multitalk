#!/usr/bin/env python3
"""
Patch for MultiTalk attention.py to use xformers instead of flash attention.
"""

def create_attention_patch():
    """
    Create a complete replacement for the flash_attention function.
    """
    return '''
def flash_attention(q, k, v, dropout_p=0.0, causal=False, version=2):
    """
    Fallback attention implementation using PyTorch's scaled_dot_product_attention.
    This replaces flash attention when it's not available.
    """
    import torch
    import torch.nn.functional as F
    
    # Use PyTorch's efficient attention implementation
    # This is available in PyTorch 2.0+ and provides good performance
    if hasattr(F, 'scaled_dot_product_attention'):
        # Convert to the expected format for scaled_dot_product_attention
        # Input shapes: [batch_size, num_heads, seq_len, head_dim]
        
        # Apply dropout if specified
        attn_kwargs = {}
        if dropout_p > 0:
            attn_kwargs['dropout_p'] = dropout_p
        if causal:
            attn_kwargs['is_causal'] = True
            
        # Use PyTorch's optimized attention
        output = F.scaled_dot_product_attention(q, k, v, **attn_kwargs)
        return output
    else:
        # Manual attention implementation as last resort
        # Scale factor
        d_k = q.size(-1)
        scale = 1.0 / (d_k ** 0.5)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask if needed
        if causal:
            seq_len = scores.size(-1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output


# Override the check for flash attention
FLASH_ATTN_2_AVAILABLE = False
FLASH_ATTN_1_AVAILABLE = False
'''

if __name__ == "__main__":
    print(create_attention_patch())