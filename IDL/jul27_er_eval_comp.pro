
Pro jul27_er_eval_comp

  r_e=6371.e3

  data_path=expand_path('C:\Users\Robert\Documents\SUVI_Occs\ER_data\')

  file=dialog_pickfile(path=data_path,title='Select File from Observation to Be Analyzed')
  
  parsed=strsplit(file,'/',/extract)

  parsed2=strsplit(parsed[-1],'_',/extract)

  date_string=strmid(parsed2[0],0,9)

  fname_root=date_string

  fnames=file_search(data_path+'/'+date_string+'*')

  ecef_array=[]
  gse_py_array=[]
  wave_array=[]
  gse_idl_array=[]

  for ff=0,n_elements(fnames)-1 do begin

     d=mrdfits(fnames[ff],0,h)
     d=mrdfits(fnames[ff],1) 


;     er=d.max_hist_freq_er
     er=d.avg_er ;obsolete after horizon correction
     h_t=d.altitude ;obsolete after horizon correction

     er_profile=er_image_to_profile(fnames[ff])

     h_t=er_profile[*,0]

     er=er_profile[*,1]
     
     white_plots
     plot,h_t,er,/ylog,xtitle='km',ytitle='ER ',title='ER with and without (--) Curvature Correction',charsize=1.8,$
          yrange=[1e-4,1.1]
     oplot,d.altitude,d.avg_er,linestyle=2
     print,fnames[ff]
stop
     parsed=strsplit(h[-7],' ',/extract)
     ecef_x=double(parsed[1])
     parsed=strsplit(h[-6],' ',/extract)
     ecef_y=double(parsed[1])
     parsed=strsplit(h[-5],' ',/extract)
     ecef_z=double(parsed[1])

     ecef_array=[[ecef_array],[ecef_x,ecef_y,ecef_z]]
;convert to gse 
     parsed0=strsplit(h[-8],' ',/extract)
     parsed=strmid(parsed0[1],1,23)
     parsed2=strsplit(parsed,'T',/extract)
     date_string=parsed2[0]
     time_string=parsed2[1]
     parsed3=strsplit(date_string,'-',/extract)
     year=float(parsed3[0])
     month=float(parsed3[1])
     day=float(parsed3[2])
     parsed4=strsplit(time_string,':',/extract)
     hr=double(parsed4[0])
     mn=double(parsed4[1])
     sc=double(parsed4[2])+0.5

     time = date2es(month,day,year,hr,mn,sc)
                                ;  pos_gse = cxform(j2000_tle,
                                ;  'J2000', 'GSE', time)                                                      
     pos_gse = cxform(ecef_array[*,-1], 'GEO', 'GSE', time)


     gse_idl_array=[[gse_idl_array],[pos_gse] ]

     
     parsed=strsplit(h[-4],' ',/extract)
     gse_x=double(parsed[1])
     parsed=strsplit(h[-3],' ',/extract)
     gse_y=double(parsed[1])
     parsed=strsplit(h[-2],' ',/extract)
     gse_z=double(parsed[1])

     gse_py_array=[[gse_py_array],[gse_x,gse_y,gse_z]]

     parsed=strsplit(fnames[ff],'_',/extract)
     parsed2=strsplit(parsed[-1],'e',/extract)
     parsed3=strsplit(parsed2[-1],'.',/extract)
     wave=float(parsed3[0])
     wave_tag='l_'+parsed3[0]+'_'+strcompress(string(ff),/remove_a)

     wave_array=[wave_array,wave_tag]

     if ff eq 0 then er_st=create_struct(wave_tag,[transpose(h_t),transpose(er)]) $
                           else er_st=create_struct(wave_tag,[transpose(h_t),transpose(er)],er_st)


  end


  alt_diff=sqrt(gse_idl_array[1,*]^2+gse_idl_array[2,*]) - sqrt(gse_py_array[1,*]^2+gse_py_array[2,*] )

;  line_desc=['___','..','--','_.','_..','__ __']

  color_lab=['black','red','green','blue','orange','purple','rust','light blue']

  color_num=[0,233,145,64,205,32,213,89]

white_plots
print,'Selection Key'
  for nn=0,n_elements(wave_array)-1 do begin

     s=execute('er_plot=er_st.'+wave_array[nn])
     
     if nn eq 0 then plot,er_plot[0,*],er_plot[1,*],charsize=2 else $
        oplot, er_plot[0,*],er_plot[1,*], color=color_num[nn]

     print,string(nn)+':   '+wave_array[nn]+'  '+color_lab[nn]



  end

;  print,'Enter Index for Best 171 or 195 ER.'
  total_ind=''
  read,total_ind,prompt='Enter Index for Best 171 or 195 ER.'

  comp_ind=''
  read,comp_ind,prompt='Enter Index for Best 284 or 304 ER.'


  parsed=strsplit(wave_array[total_ind],'_',/extract)
  tot_wv=float(parsed[1])/10.

  parsed=strsplit(wave_array[comp_ind],'_',/extract)
  comp_wv=float(parsed[1])/10.

  s=execute('tot_er=er_st.'+wave_array[total_ind])

  tot_er[1,*]=smooth(tot_er[1,*],10,/edge_trun)

  ind=where(tot_er[1,*] ge 0.001 and tot_er[1,*] le 0.999 )
  tot_er=tot_er[*,ind]

  s=execute('comp_er=er_st.'+wave_array[comp_ind])

  comp_er[1,*]=smooth(comp_er[1,*],10,/edge_trun)

  ind=where(comp_er[1,*] ge 0.001 and comp_er[1,*] le 0.999 )
  comp_er=comp_er[*,ind]


  restore,'$idl_code/lyra_occultations/data/cross_sections/photon_cross_sections.sav'
  
  cs_wv=photo.angstroms/10.
  cs_o=photo.o3p.xsection[0,*]
  cs_n2=photo.n2.xsection[0,*]

  cs_o_t=interpol(cs_o,cs_wv,tot_wv)
  cs_o_c=interpol(cs_o,cs_wv,comp_wv)

  cs_n2_t=interpol(cs_n2,cs_wv,tot_wv)
  cs_n2_c=interpol(cs_n2,cs_wv,comp_wv)

  n_col_t=-1./cs_o_t * alog(tot_er[1,*])


  ind=where(tot_er[0,*] le max(comp_er[0,*]) and tot_er[0,*] ge min(comp_er[0,*]) )
  comp_er_int=interpol(comp_er[1,*],comp_er[0,*],tot_er[0,ind])


  n_col_o=(alog(comp_er_int)+n_col_t[ind]*cs_n2_c  )/( cs_n2_c - cs_o_c  )

  ind2=where(n_col_o gt 0.)

  n_col_o=[tot_er[0,ind[ind2]],n_col_o[0,ind2] ]

  n_col_t=[tot_er[0,*],n_col_t]



;  save,n_col_t,filename='/Users/thiemann/Desktop/temp/temp_n_col.sav'

;find number density and plot                                                                       
  ind=where(n_col_t[0,*] ge 160 and  n_col_t[0,*] le 320)
 
  d_t=n_num_abel_suvi(n_col_t[0,ind]+r_e/1e3,n_col_t[1,ind])
  ind=where(n_col_o[0,*] ge 160 and  n_col_o[0,*] le 320)

;  save,d_t,filename='/Users/thiemann/Desktop/temp/temp_n_den.sav'  
  
  d_o=n_num_abel_suvi(n_col_o[0,ind]+r_e/1e3,n_col_o[1,ind])

  d_n2=d_o

  d_t_int=interpol(d_t[1,*],d_t[0,*],d_o[0,*])

  d_n2[1,*]=d_t_int-d_o[1,*]
  

  p1=plot(d_o[1,*],d_o[0,*]-r_e/1e3,/xlog,color='red',ytitle='Altitude (km)',xtitle='Density (cm!e-3!n)',yrange=[160,300])
  p1=plot(d_n2[1,*],d_n2[0,*]-r_e/1e3,/over,color='blue')
  p1=plot(d_t[1,*],d_t[0,*]-r_e/1e3,/over)

  readcol,'$idl_code/SUVI_occultations/data/msis_091519.txt',m_z,m_o,m_n2,m_t
  p1=plot(m_o,m_z,linestyle=2,color='red',/over)
  p1=plot(m_n2,m_z,linestyle=2,color='blue',/over)

  ;find thermospheric temperature

  n=d_t[1,*]*(100.)^3. ;change to SI units

  ind=where(d_t[0,*] ge 270.+r_e/1e3 and d_t[0,*] le 300 +r_e/1e3) ;define fit range

  coefs=poly_fit((d_t[0,ind]-r_e/1e3-220)*1000.,alog(n[ind]),1) ;do power law fit to find scale height
  
  h=-1.*1./coefs[1] ;scale height is inverse of fit slope

  m_o=1;16.*get_constants(/amu) ;get oxygen mass
  
  k=1;get_constants(/k) ;boltzmann's constant

  g=9.8*(r_e/(r_e+250.))^2. ;gravity at 250 km

  ;H=kT/mg

  ;mgH/k=T

  t=m_o*g*h/k ;temperature found from scale height
  
  print,'T at 275 km is:'
  print,t

  t_285=t

  fname='/Users/thiemann/Documents/Working/IDL/git/evesci2/SUVI_occultations/data/retrieved_densities/'+'suvi_den_'+fname_root+'v2.sav'

  save,d_o,d_n2,d_t,comp_wv,tot_wv,t_285,n_col_o,n_col_t,gse_py_array,gse_idl_array,wave_array,$
       filename=fname



End
