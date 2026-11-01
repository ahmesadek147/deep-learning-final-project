import json, tempfile, unittest
from pathlib import Path
import torch
from src.training.p4_continue import prepare, verify_resume, digest, file_hash

class Checks(unittest.TestCase):
 def setUp(self):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
  self.s=self.root/'original';self.s.mkdir();self.d=self.root/'new'
  (self.root/'manifest.json').write_text('{}')
  self.c={'model':{'name':'residual_cnn','pretrained':False},'data':{'dataset_name':'Multimodal-Fatima/CUB_train','split_strategy':'stratified_group','image_size':224,'split_manifest':'manifest.json'},'training':{'epochs':80,'monitor':'val_macro_f1','early_stopping':True,'patience':20}}
  rows=[{'epoch':i,'val_macro_f1':.2377 if i==79 else .2,'val_accuracy':.25,'val_loss':3.8} for i in range(1,81)]
  self.l={'epoch':80,'model_state_dict':{'w':torch.ones(2)},'optimizer_state_dict':{'moment':torch.ones(2),'lr':.0005},'scheduler_state_dict':{'best':.2377},'scaler_state_dict':{'scale':1024},'rng':{'torch':torch.get_rng_state()},'history':rows,'best_epoch':79,'best_key':(.2377,.25,-3.8),'wait':1,'config_hash':digest(self.c)}
  self.b={'epoch':79,'model_state_dict':{'w':torch.zeros(2)},'metadata':{'model_config':self.c['model'],'image_size':224},'metrics':rows[78]};self.save()
 def save(self):
  (self.s/'config.json').write_text(json.dumps(self.c));torch.save(self.l,self.s/'last.pth');torch.save(self.b,self.s/'best_model.pth')
 def make(self): return prepare(self.s,self.d,120,self.root,{})
 def test_preserved_state_and_source(self):
  before={p.name:file_hash(p) for p in self.s.iterdir()};c,p=self.make();l=torch.load(self.d/'last.pth',weights_only=True)
  self.assertEqual(l['config_hash'],digest(c));self.assertEqual(c['training']['epochs'],120);self.assertEqual(l['epoch'],80)
  for k in ['scheduler_state_dict','scaler_state_dict','history','best_key','wait']:self.assertEqual(l[k],self.l[k])
  self.assertTrue(torch.equal(l['optimizer_state_dict']['moment'],self.l['optimizer_state_dict']['moment']))
  self.assertTrue(torch.equal(l['rng']['torch'],self.l['rng']['torch']))
  self.assertEqual(file_hash(self.d/'best_model.pth'),before['best_model.pth']);self.assertEqual(before,{p.name:file_hash(p) for p in self.s.iterdir()});verify_resume(self.d,self.s,120,{})
 def test_existing_destination(self):
  self.d.mkdir()
  with self.assertRaises(FileExistsError):self.make()
 def test_config_mismatch(self):
  self.c['training']['epochs']=90;self.save()
  with self.assertRaisesRegex(ValueError,'does not match'):self.make()
 def test_stopped(self):
  self.l['wait']=20;self.save()
  with self.assertRaisesRegex(ValueError,'early stopping'):self.make()
 def test_missing_state(self):
  del self.l['rng'];self.save()
  with self.assertRaisesRegex(ValueError,'Incomplete'):self.make()
 def test_changed_manifest(self):
  self.make();(self.root/'manifest.json').write_text('changed')
  with self.assertRaisesRegex(ValueError,'manifest changed'):verify_resume(self.d,self.s,120,{})
 def test_overlap(self):
  with self.assertRaisesRegex(ValueError,'separate'):prepare(self.s,self.s,120,self.root,{})

if __name__=='__main__':unittest.main()
