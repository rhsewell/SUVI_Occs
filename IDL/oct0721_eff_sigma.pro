Pro oct0721_eff_sigma

  s=read_csv('/Users/thiemann/Documents/Working/IDL/git/evesci2/SUVI_occultations/data/solar_spectra/fism2_09152019.csv')

  irr_wv=s.field2
  irr=s.field3

  ;find weights for each channel

  restore,'$idl_code//SUVI_occultations/data/response_functions/suvi_resp_171_g16.sav'
  w_171=irr*interpol(resp,wv,irr_wv)
  ind=where(irr_wv le 100)
  w_171=w_171[ind]/total(w_171[ind])

  restore,'$idl_code//SUVI_occultations/data/response_functions/suvi_resp_195_g16.sav'
  w_195=irr*interpol(resp,wv,irr_wv)
  ind=where(irr_wv le 100)
  w_195=w_195[ind]/total(w_195[ind])

  restore,'$idl_code//SUVI_occultations/data/response_functions/suvi_resp_284_g16.sav'
  w_284=irr*interpol(resp,wv,irr_wv)
  ind=where(irr_wv le 100)
  w_284=w_284[ind]/total(w_284[ind])

  restore,'$idl_code//SUVI_occultations/data/response_functions/suvi_resp_304_g16.sav'
  w_304=irr*interpol(resp,wv,irr_wv)
  ind=where(irr_wv le 100)
  w_304=w_304[ind]/total(w_304[ind])


  ;find effective cross sections

  readcol,'$data/cross_sections/n2_chan.txt',wv_chan,n2_chan    

  sigma_n2=interpol(n2_chan,wv_chan,irr_wv[ind])

  cs_n2_171=total(w_171*sigma_n2)
  cs_n2_195=total(w_195*sigma_n2)
  cs_n2_284=total(w_284*sigma_n2)
  cs_n2_304=total(w_304*sigma_n2)

  restore,'$idl_code/lyra_occultations/data/cross_sections/photon_cross_sections.sav'

  cs_wv=photo.angstroms/10.
  cs_o=photo.o3p.xsection[0,*]

  sigma_o=interpol(cs_o,cs_wv,irr_wv[ind])


;  d=read_csv('/Users/thiemann/Documents/data/cross_sections/sigma_o_fennelly_torr.csv')

;  ind2=where(irr_wv ge 10 and irr_Wv le 40)

;  sigma_o=interpol(d.field2*1e-18,d.field1,irr_wv[ind2])
;  w_171=interpol(w_171,irr_Wv[ind],irr_wv[ind2])
; w_195=interpol(w_195,irr_Wv[ind],irr_wv[ind2])
; w_284=interpol(w_284,irr_Wv[ind],irr_wv[ind2])
; w_304=interpol(w_304,irr_Wv[ind],irr_wv[ind2])

; w_171=w_171/total(w_171)
; w_195=w_195/total(w_195)
; w_284=w_284/total(w_284)
; w_304=w_304/total(w_304)


  cs_o_171=total(w_171*sigma_o)
  cs_o_195=total(w_195*sigma_o)
  cs_o_284=total(w_284*sigma_o)
  cs_o_304=total(w_304*sigma_o)

  save,filename='$idl_code/SUVI_occultations/data/cross_sections/suvi_effective_cross_sections.sav',$
  cs_n2_171,cs_n2_195,cs_n2_284,cs_n2_304,cs_o_171,cs_o_195,cs_o_284,cs_o_304




End
