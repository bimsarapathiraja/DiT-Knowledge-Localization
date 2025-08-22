# Import required modules for Flux attention processors
from enum import Enum, auto        # For creating enumerations with auto-generated values
import math                        # Mathematical functions (for sqrt in scaled dot product attention)
import torch                       # PyTorch tensor operations
from typing import Optional        # Type hints for optional parameters
import torch.nn.functional as F    # PyTorch functional operations (scaled_dot_product_attention)
from diffusers.models.attention_processor import Attention  # Base attention processor class


class CachingAttnProcessor:
    def __init__(self, idx):
        self.idx = idx
        self.attention_maps = []
        self.values = []
        self.out_norms = []
        self.in_norms = []
    
    def clear_maps(self):
        self.attention_maps = []
        self.values = []
        self.out_norms = []
        self.in_norms = []

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        with torch.no_grad():
            self.in_norms.append(hidden_states[1].detach().cpu())

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attention_maps.append(attention_probs.detach().cpu())
        self.values.append(value.detach().cpu())

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        with torch.no_grad():
            self.out_norms.append(torch.norm(hidden_states[1], dim=1).detach().cpu())

        return hidden_states


class EmbeddingModifierAttnProcessor:
    def __init__(self, idx, modify_emb=False, print_log=False):
        self.idx = idx
        self.modify_emb = modify_emb
        self.print_log = print_log
    
    def set_clean_emb_and_mask(self, clean_emb, clean_mask):
        self.clean_emb = clean_emb
        self.clean_mask = clean_mask

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        clean_emb = None,
        clean_mask = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        
        encoder_hidden_states = encoder_hidden_states.clone()
        attention_mask = attention_mask.clone()

        if self.modify_emb:
            if clean_emb is None and self.clean_emb is not None:
                clean_emb = self.clean_emb
                clean_mask = self.clean_mask
            assert clean_emb is not None and clean_mask is not None
            
            encoder_hidden_states[1] = clean_emb
            attention_mask[1] = clean_mask
            if self.print_log:
                print(f"Replaced for block {self.idx}")


        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

class FluxCachingAttnProcessor:
    def __init__(self, idx, max_text_tokens_len=512):
        self.idx = idx
        self.max_text_tokens_len = max_text_tokens_len

        self.attention_maps = []
        self.values = []
        self.out_norms = []
        self.in_norms = []
    
    def clear_maps(self):
        self.attention_maps = []
        self.values = []
        self.out_norms = []
        self.in_norms = []
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        assert attention_mask is None, "Attention mask is not supported in this processor."
        # assert encoder_hidden_states is not None, "Encoder hidden states are required in this processor."

        with torch.no_grad():
            self.in_norms.append(hidden_states[0].detach().cpu())

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        scale_factor = 1 / math.sqrt(query.size(-1))
        attention_probs = query @ key.transpose(-2, -1) * scale_factor
        attention_probs = torch.softmax(attention_probs, dim=-1)

        self.attention_maps.append(attention_probs[:, :, self.max_text_tokens_len:, :self.max_text_tokens_len].detach().cpu())    
        self.values.append(value[:, :, :self.max_text_tokens_len, :].detach().cpu())

        hidden_states = attention_probs @ value
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            with torch.no_grad():
                self.out_norms.append(torch.norm(hidden_states[0], dim=1).detach().cpu())

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
        

class FluxAttnContCalculatorProcessor:
    """
    Custom attention processor for measuring attention contributions in Flux transformer blocks.
    
    This processor calculates how much each attention layer contributes to generating
    specific tokens (representing target knowledge). It accumulates these contributions
    across multiple forward passes to identify the most influential blocks.
    """
    def __init__(self, token_indices_for_attn_cont_calc, max_text_tokens_len=512):
        """
        Initialize the attention contribution calculator.
        
        Args:
            token_indices_for_attn_cont_calc: Indices of tokens to measure contributions for
            max_text_tokens_len: Maximum length of text token sequence (default 512 for Flux)
        """
        # Maximum length of text tokens in the sequence (Flux uses up to 512)
        self.max_text_tokens_len = max_text_tokens_len

        # Token indices to focus on when calculating attention contributions
        # These represent the target knowledge tokens we want to measure
        self.token_indices_for_attn_cont_calc = token_indices_for_attn_cont_calc

        # Accumulated attention contribution score across all forward passes
        self.attn_contribution = 0.
        # Number of times the attention contribution was updated (for averaging)
        self.attn_contribution_update_count = 0
    
    def calc_attn_cont(self, attention_probs, value, query, attn):
        """
        Calculate attention contribution for the target tokens.
        
        This method computes how much the attention mechanism contributes to
        generating the target knowledge by measuring the norm of the output
        when focusing only on target tokens.
        
        Args:
            attention_probs: Attention probability matrix [batch, heads, seq_len, seq_len]
            value: Value matrix [batch, heads, seq_len, head_dim] 
            query: Query matrix [batch, heads, seq_len, head_dim]
            attn: Attention layer object with output projection
            
        Returns:
            float: Attention contribution score for this forward pass
        """
        # Ensure we have valid token indices to calculate contributions for
        assert isinstance(self.token_indices_for_attn_cont_calc, list)
        assert len(self.token_indices_for_attn_cont_calc) > 0

        # Extract attention from image tokens to text tokens 
        # Shape: [heads, image_seq_len, text_seq_len] -> [heads, image_seq_len, target_tokens]
        m = attention_probs[:, :, self.max_text_tokens_len:, :self.max_text_tokens_len].detach().clone()
        m = m[0, :, :, self.token_indices_for_attn_cont_calc]  # Focus only on target token columns
        
        # Extract values corresponding to target tokens
        # Shape: [heads, text_seq_len, head_dim] -> [heads, target_tokens, head_dim]
        v = value[:, :, :self.max_text_tokens_len, :].detach().clone()
        v = v[0, :, self.token_indices_for_attn_cont_calc, :]  # Focus only on target token rows
        
        # Compute attention-weighted values: attention_to_targets * target_values
        # Then reshape for output projection: [heads, image_seq_len, head_dim] -> [image_seq_len, total_dim]
        o = (m @ v).transpose(0, 1).reshape(m.shape[1], v.shape[0] * v.shape[2]) # (24, 1024, 128) -> (1024, 24*128)
        o = o.to(query.dtype)  # Ensure consistent dtype with model
        
        # Apply output projection if it exists (linear layer)
        if attn.to_out:
            o = attn.to_out[0](o) # (1024, 24*128) -> (1024, 24*128)
        
        # Calculate L2 norm across feature dimension and take mean across spatial dimension
        # This measures the magnitude of the output influenced by target tokens
        attn_cont = torch.norm(o.to(torch.float32), dim=1).mean().item()

        return attn_cont

    def update_attn_cont(self, attn_cont):
        """
        Accumulate attention contribution scores across multiple forward passes.
        
        Args:
            attn_cont: Attention contribution score from current forward pass
        """
        # Add current contribution to running total
        self.attn_contribution += attn_cont
        # Increment update counter for later averaging
        self.attn_contribution_update_count += 1

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass of attention with contribution calculation.
        
        This performs standard Flux attention computation while measuring and accumulating
        attention contributions to target tokens.
        
        Args:
            attn: Attention layer instance with projection weights
            hidden_states: Image token embeddings [batch, image_seq_len, dim]
            encoder_hidden_states: Text token embeddings [batch, text_seq_len, dim] or None
            attention_mask: Attention mask (not used in Flux)
            image_rotary_emb: Rotary position embeddings for image tokens
            
        Returns:
            torch.FloatTensor: Processed hidden states, optionally with encoder states
        """
        # Flux doesn't use attention masks - ensure none provided
        assert attention_mask is None, "Attention mask is not supported in this processor."
        # Encoder hidden states are optional in Flux (single vs multi-modal blocks)
        # assert encoder_hidden_states is not None, "Encoder hidden states are required in this processor."

        # Get batch size from appropriate input tensor
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # Generate query, key, and value projections from image tokens (`sample` projections)
        query = attn.to_q(hidden_states)  # Project image tokens to query space
        key = attn.to_k(hidden_states)    # Project image tokens to key space  
        value = attn.to_v(hidden_states)  # Project image tokens to value space

        # Calculate dimensions for multi-head attention
        inner_dim = key.shape[-1]           # Total dimension across all heads
        head_dim = inner_dim // attn.heads  # Dimension per attention head

        # Reshape tensors for multi-head attention: [batch, seq_len, dim] -> [batch, heads, seq_len, head_dim]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply query normalization if present (improves training stability)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        # Apply key normalization if present
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Handle encoder hidden states for multi-modal attention (text + image)
        # Single transformer blocks don't use encoder_hidden_states, only multi-modal blocks do
        if encoder_hidden_states is not None:
            # Generate additional query, key, value projections from text tokens (`context` projections)
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)  # Text to query
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)    # Text to key
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)  # Text to value

            # Reshape encoder projections for multi-head attention
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            # Apply normalization to encoder projections if present
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Concatenate encoder (text) and image projections along sequence dimension
            # This creates joint text+image attention: [batch, heads, text_seq+img_seq, head_dim]
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # Apply rotary position embeddings if provided (for spatial relationships)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            # Apply rotary embeddings to query and key (not value)
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Compute scaled dot-product attention manually
        scale_factor = 1 / math.sqrt(query.size(-1))                    # Scale factor for numerical stability
        attention_probs = query @ key.transpose(-2, -1) * scale_factor  # Compute attention scores
        attention_probs = torch.softmax(attention_probs, dim=-1)        # Convert scores to probabilities

        # **KEY STEP**: Calculate and accumulate attention contributions to target tokens
        self.update_attn_cont(self.calc_attn_cont(attention_probs, value, query, attn))

        # Apply attention to values to get final hidden states
        hidden_states = attention_probs @ value  # Apply attention weights to values
        # Reshape back to original format: [batch, heads, seq_len, head_dim] -> [batch, seq_len, total_dim]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)  # Ensure consistent dtype

        # Handle outputs for multi-modal vs single transformer blocks
        if encoder_hidden_states is not None:
            # Split back into encoder (text) and decoder (image) components
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],  # Text portion
                hidden_states[:, encoder_hidden_states.shape[1] :],  # Image portion
            )

            # Apply output projections to image hidden states
            hidden_states = attn.to_out[0](hidden_states)  # Linear projection
            hidden_states = attn.to_out[1](hidden_states)  # Dropout

            # Apply output projection to encoder hidden states
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # Return both image and text hidden states for multi-modal blocks
            return hidden_states, encoder_hidden_states
        else:
            # For single transformer blocks, only return image hidden states
            return hidden_states


class FluxEmbeddingModifierAttnProcessor:
    """
    Attention processor for modifying encoder hidden states during knowledge intervention.
    
    This processor enables selective replacement of text embeddings during the diffusion process.
    It's used to implement knowledge intervention by saving "clean" text embeddings (without
    target knowledge) and replacing the original embeddings at specific blocks.
    """
    
    class ProcessorMode(Enum):
        """
        Enumeration of different processing modes for the embedding modifier.
        
        NONE: Normal attention processing without any modifications
        SAVE_ENCODER_HIDDEN_STATES: Save encoder states during "clean" forward pass
        REPLACE_ENCODER_HIDDEN_STATES: Replace encoder states with saved clean states
        """
        NONE = auto()                        # No special processing
        SAVE_ENCODER_HIDDEN_STATES = auto()  # Save mode for clean pass
        REPLACE_ENCODER_HIDDEN_STATES = auto()  # Replace mode for intervention

    def __init__(self, mode: ProcessorMode = ProcessorMode.NONE):
        """
        Initialize the embedding modifier processor.
        
        Args:
            mode: Initial processing mode (default is NONE for normal attention)
        """
        # Ensure PyTorch 2.0+ for scaled_dot_product_attention support
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        # Current processing mode (can be changed dynamically)
        self.mode = mode
        # Cached encoder hidden states from the clean forward pass
        self.saved_encoder_hidden_states = None
    
    def clear_cache(self):
        """Clear cached encoder hidden states to free memory."""
        self.saved_encoder_hidden_states = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass with optional encoder hidden state modification for knowledge intervention.
        
        This method performs Flux attention while conditionally saving or replacing encoder
        hidden states based on the current processor mode. This enables knowledge intervention
        by swapping text embeddings during generation.
        
        Args:
            attn: Attention layer instance
            hidden_states: Image token embeddings [batch, image_seq_len, dim]
            encoder_hidden_states: Text token embeddings [batch, text_seq_len, dim] or None
            attention_mask: Attention mask (unused in Flux)
            image_rotary_emb: Rotary position embeddings for image tokens
            
        Returns:
            torch.FloatTensor: Processed hidden states, optionally with modified encoder states
        """
        # Handle encoder hidden state modification based on current mode
        if encoder_hidden_states is None and self.mode == self.ProcessorMode.SAVE_ENCODER_HIDDEN_STATES:
            # For single transformer blocks: save first 512 tokens (text portion) from hidden_states
            # This happens during the "clean" forward pass to capture embeddings without target knowledge
            self.saved_encoder_hidden_states = hidden_states[:, :512, :].clone()
        elif encoder_hidden_states is None and self.mode == self.ProcessorMode.REPLACE_ENCODER_HIDDEN_STATES:
            # For single transformer blocks: replace first 512 tokens with saved clean embeddings
            # This implements the knowledge intervention by using clean text embeddings
            hidden_states[:, :512, :] = self.saved_encoder_hidden_states
            
        # Get batch size for tensor reshaping
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # Generate query, key, and value projections from image tokens
        query = attn.to_q(hidden_states)  # Project image tokens to query space
        key = attn.to_k(hidden_states)    # Project image tokens to key space
        value = attn.to_v(hidden_states)  # Project image tokens to value space

        # Calculate multi-head attention dimensions
        inner_dim = key.shape[-1]           # Total feature dimension
        head_dim = inner_dim // attn.heads  # Feature dimension per attention head

        # Reshape for multi-head attention: [batch, seq_len, dim] -> [batch, heads, seq_len, head_dim]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply normalization to query and key if present
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Handle multi-modal attention with encoder hidden states (text tokens)
        # Single transformer blocks don't use encoder_hidden_states, only multi-modal blocks do
        if encoder_hidden_states is not None:
            # Handle encoder state modification for multi-modal blocks
            if self.mode == self.ProcessorMode.SAVE_ENCODER_HIDDEN_STATES:
                # Save clean text embeddings during the clean forward pass
                self.saved_encoder_hidden_states = encoder_hidden_states.clone()
            elif self.mode == self.ProcessorMode.REPLACE_ENCODER_HIDDEN_STATES:
                # Replace current text embeddings with saved clean ones for intervention
                encoder_hidden_states = self.saved_encoder_hidden_states

            # Generate additional projections from text tokens for multi-modal attention
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)  # Text to query
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)    # Text to key
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)  # Text to value

            # Reshape encoder projections for multi-head attention
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            # Apply normalization to encoder projections if present
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Concatenate text and image projections for joint attention
            # Creates unified text+image sequence: [batch, heads, text_seq+img_seq, head_dim]
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # Apply rotary position embeddings if provided
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            # Apply rotary embeddings for spatial position encoding
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Compute efficient scaled dot-product attention using PyTorch's optimized function
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, 
            attn_mask=attention_mask,  # Usually None for Flux
            dropout_p=0.0,             # No dropout during inference
            is_causal=False            # Bidirectional attention
        )
        
        # Reshape back to standard format: [batch, heads, seq_len, head_dim] -> [batch, seq_len, total_dim]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)  # Maintain dtype consistency

        # Handle output processing for different block types
        if encoder_hidden_states is not None:
            # Multi-modal block: split output back into text and image components
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],  # Text portion
                hidden_states[:, encoder_hidden_states.shape[1] :],  # Image portion
            )

            # Apply output projections to image hidden states
            hidden_states = attn.to_out[0](hidden_states)  # Linear projection
            hidden_states = attn.to_out[1](hidden_states)  # Dropout layer

            # Apply output projection to text hidden states
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # Return both text and image hidden states
            return hidden_states, encoder_hidden_states
        else:
            # Single transformer block: only return processed hidden states
            return hidden_states
