import tensorflow as tf
import numpy as np

ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3

beam_size = 12
min_dec_steps = 1
max_dec_steps = 100

class Hypothesis(object):

  def __init__(self, tokens, log_probs, distract_state, state, attn_dists, p_ptrs):
   
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.distract_state = distract_state
    self.attn_dists = attn_dists
    self.p_ptrs = p_ptrs

  def extend(self, token, log_prob, distract_state, state, attn_dist, p_ptr):    
    return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      distract_state = distract_state,
                      state = state,
                      attn_dists = self.attn_dists + [attn_dist],
                      p_ptrs = self.p_ptrs + [p_ptr])

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return sum(self.log_probs)

  @property
  def avg_log_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return self.log_prob / len(self.tokens)


def run_beam_search(sess, model, vocab, batch):
  vocab_size = len(vocab[0])
  # Run the encoder to get the encoder hidden states and decoder initial state
  en_doc_outputs, en_query_outputs, distract_state, dec_in_state = model.run_encoder(sess, batch)
  # dec_in_state is a LSTMStateTuple
  # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

  # Initialize beam_size-many hyptheses
  hyps = [Hypothesis(tokens=[ID_GO],
                     log_probs=[0.0],
                     distract_state=distract_state,
                     state=dec_in_state,
                     attn_dists=[],
                     p_ptrs=[],
                     ) for i in range(beam_size)]
  results = [] # this will contain finished hypotheses (those that have emitted the [STOP] token)

  steps = 0
  while steps < max_dec_steps and len(results) < beam_size:
    latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis
    latest_tokens = [t if t in range(vocab_size) else ID_UNK for t in latest_tokens] # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
    states = [h.state for h in hyps] # list of current decoder states of the hypotheses
    distract_states = [h.distract_state for h in hyps] # list of current decoder states of the hypotheses

    # Run one step of the decoder to get the new info
    (topk_ids, topk_log_probs, new_distract_states, new_states, attn_dists, p_ptrs) = model.decode_onestep(sess=sess,
                        batch=batch,
                        latest_tokens=latest_tokens,
                        en_doc_outputs=en_doc_outputs, 
                        en_query_outputs=en_query_outputs,
                        distract_states=distract_states,
                        dec_init_states=states)
    #print ("TOPK SHAPE: ", np.shape(topk_ids))
    # Extend each hypothesis and collect them all in all_hyps
    all_hyps = []
    num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
    for i in range(num_orig_hyps):
      h, new_distract_state, new_state, attn_dist, p_ptr = hyps[i], new_distract_states[i], new_states[i], attn_dists[i], p_ptrs[i]  # take the ith hypothesis and new decoder state info
      for j in range(beam_size * 2):  # for each of the top 2*beam_size hyps:
        # Extend the ith hypothesis with the jth option
        new_hyp = h.extend(token=topk_ids[i,0, j],
                           log_prob=topk_log_probs[i,0, j],
                           distract_state=new_distract_state,
                           state=new_state,
                           attn_dist=attn_dist,
                           p_ptr=p_ptr)
        all_hyps.append(new_hyp)

    # Filter and collect any hypotheses that have produced the end token.
    hyps = [] # will contain hypotheses for the next step
    for h in sort_hyps(all_hyps): # in order of most likely h
      if h.latest_token == ID_EOS: # if stop token is reached...
        # If this hypothesis is sufficiently long, put in results. Otherwise discard.
        if steps >= min_dec_steps:
          results.append(h)
      else: # hasn't reached stop token, so continue to extend this hypothesis
        hyps.append(h)
      if len(hyps) == beam_size or len(results) == beam_size:
        # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
        break

    steps += 1

  # At this point, either we've got beam_size results, or we've reached maximum decoder steps

  if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    results = hyps

  # Sort hypotheses by average log probability
  hyps_sorted = sort_hyps(results)

  # Return the hypothesis with highest average log prob
  return hyps_sorted[0]

def sort_hyps(hyps):
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)