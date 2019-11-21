"""Create model function for estimator."""

# pylint: disable=g-import-not-at-top
try:
  from google3.experimental.users.zihangd.pretrain.dae_model_func_builder \
      import dae_loss, dae_joint_loss
  from google3.experimental.users.zihangd.pretrain.encdec_model_func_builder \
      import joint_loss, encdec_loss
  from google3.experimental.users.zihangd.pretrain.mlm_model_func_builder \
      import mlm_loss, get_lm_loss, electra_loss
  from google3.experimental.users.zihangd.pretrain.mass_model_func_builder \
      import mass_loss
except ImportError:
  from dae_model_func_builder import dae_loss, dae_joint_loss
  from seq2seq_model_func_builder import joint_loss, encdec_loss#, joint_rel_attn_loss
  from mlm_model_func_builder import mlm_loss, get_lm_loss, electra_loss
  from mass_model_func_builder import mass_loss
  # pylint: enable=g-import-not-at-top
