
Pro suvi_n_col

  r_e=6371.e3

  data_path=expand_path('C:\\Users\\Robert\\Documents\\GitHub\\SUVI_Occs\\Retrieval_Data\\')

  file=dialog_pickfile(path=data_path,title='Select File from Observation to Be Analyzed')
  
  parsed=strsplit(file,'\',/extract)

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
     
     ;white_plots
;     plot,h_t,er,/ylog,xtitle='km',ytitle='ER ',title='ER with and without (--) Curvature Correction',charsize=1.8,$
 ;         yrange=[1e-4,1.1]
 ;    oplot,d.altitude,d.avg_er,linestyle=2
 ;    print,fnames[ff]
;stop
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

;white_plots
print,'Selection Key'
  for nn=0,n_elements(wave_array)-1 do begin

     s=execute('er_plot=er_st.'+wave_array[nn])
     
     if nn eq 0 then p1=plot(er_plot[0,*],er_plot[1,*]) else $
        p2=plot(er_plot[0,*],er_plot[1,*], color=color_lab[nn],/over)

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

  tot_er[1,*]=smooth(tot_er[1,*],1,/edge_trun)

  ind=where(tot_er[1,*] ge 0.001 and tot_er[1,*] le 0.999 )
  tot_er=tot_er[*,ind]

  s=execute('comp_er=er_st.'+wave_array[comp_ind])

  comp_er[1,*]=smooth(comp_er[1,*],1,/edge_trun)

  ind=where(comp_er[1,*] ge 0.001 and comp_er[1,*] le 0.999 )
  comp_er=comp_er[*,ind]


  ;restore,'C:\Users\Robert\Documents\SUVI_Occs\photon_cross_sections.sav'
  
  ;cs_wv=photo.angstroms/10.
  ;cs_o=photo.o3p.xsection[0,*]
  ;cs_n2=photo.n2.xsection[0,*]

  ;readcol,'$data/cross_sections/n2_chan.txt',wv_chan,n2_chan

  ;cs_n2=interpol(n2_chan,wv_chan,cs_wv)


  cs_wv=[17.106,19.5,28.415,30.378]

  cs_o=[3.26,4.25,7.08,7.7]*1e-18
  cs_n2=[4.3,5.83,10.9,11.1]*1e-18


  cs_o_t=interpol(cs_o,cs_wv,tot_wv)
  cs_o_c=interpol(cs_o,cs_wv,comp_wv)

  cs_n2_t=interpol(cs_n2,cs_wv,tot_wv)
  cs_n2_c=interpol(cs_n2,cs_wv,comp_wv)

  n_col_t=-1./cs_o_t * alog(tot_er[1,*])


  ind=where(tot_er[0,*] le max(comp_er[0,*]) and tot_er[0,*] ge min(comp_er[0,*]) )
  comp_er_int=interpol(comp_er[1,*],comp_er[0,*],tot_er[0,ind])

  log_n=smooth(alog(n_col_t),10,/edge_tru)

  n_col_t=exp(log_n)

  n_col_o=(alog(comp_er_int)+n_col_t[ind]*cs_n2_c  )/( cs_n2_c - cs_o_c  )

  log_n=smooth(alog(n_col_o),10,/edge_tru)

  n_col_o=exp(log_n)

  ind2=where(n_col_o gt 0.)

  n_col_o=[tot_er[0,ind[ind2]],n_col_o[0,ind2] ]

  n_col_t=[tot_er[0,*],n_col_t]

  ind=where(n_col_t[0,*] le max(n_col_o[0,*]) and n_col_t[0,*] ge min(n_col_o[0,*]) )

  n_col_t_int=interpol(n_col_t[1,ind],n_col_t[0,ind],n_col_o[0,*])

  n_col_n2=[n_col_o[0,*],n_col_t_int-n_col_o[1,*] ]

  ind=where(n_col_o[0,*] ge 220 and n_col_o[0,*] le 300)
  coefs_o=poly_fit(n_col_o[0,ind]-n_col_o[0,0],alog(n_col_o[1,ind]),1)

  ind=where(n_col_n2[0,*] ge 220 and n_col_n2[0,*] le 280)
  coefs_n2=poly_fit(n_col_n2[0,ind]-n_col_n2[0,0],alog(n_col_n2[1,ind]),1)

  
  z_fit=findgen(150)+200

  o_fit=exp(coefs_o[0])*exp(coefs_o[1]*(z_fit-n_col_o[0,0]) )
  n2_fit=exp(coefs_n2[0])*exp(coefs_n2[1]*(z_fit-n_col_n2[0,0]) )


 
  p1=plot(n_col_o[1,*],n_col_o[0,*],/xlog,color='red',ytitle='Altitude (km)',xtitle='Density (cm!e-2!n)',yrange=[160,330])

  p1=plot(n_col_n2[1,*],n_col_n2[0,*],/over,color='blue')

  p1=plot(o_fit,z_fit,/over,color='red',linestyle=':')
  p1=plot(n2_fit,z_fit,/over,color='blue',linestyle=':')

  b=''
  read,b,prompt='Press Enter to continue.'


  ;get ecef coordinates for every altitude array

  sc_ecef=[mean(ecef_array[0,*]),mean(ecef_array[1,*]),mean(ecef_array[2,*])  ]
  

  fname='/Users/thiemann/Documents/Working/IDL/git/evesci2/SUVI_occultations/data/retrieved_columns/'+'suvi_col_den_'+fname_root+'_v01r00.sav'

  n_col_o[0,*]=n_col_o[0,*]+r_e
  n_col_n2[0,*]=n_col_n2[0,*]+r_e
  r_fit=z_fit+r_e

  theta=atan(sc_ecef[2]/sc_ecef[1])

  o_gse=[0*findgen(1,n_elements(n_col_o[0,*])),n_col_o[0,*]*sin(theta),n_col_o[0,*]*cos(theta) ]*1e3
  n2_gse=[0*findgen(1,n_elements(n_col_n2[0,*])),n_col_n2[0,*]*sin(theta),n_col_n2[0,*]*cos(theta) ]*1e3
  fit_gse=[0*findgen(1,n_elements(r_fit)),transpose(r_fit)*sin(theta),transpose(r_fit)*cos(theta) ]*1e3

  time_o=findgen(n_elements(o_Gse[0,*]))
  time_o[*]=time

  time_n2=findgen(n_elements(n2_Gse[0,*]))
  time_n2[*]=time

  time_fit=findgen(n_elements(fit_Gse[0,*]))
  time_fit[*]=time


  o_ecef=cxform(o_gse,'GSE','GEO',time_o)
  n2_ecef=cxform(n2_gse,'GSE','GEO',time_n2)
  fit_ecef=cxform(fit_gse,'GSE','GEO',time_fit)

  read_me='n_col_o and n_col_o2 are retrieved column densities versus radial distance from earth center [km,cm^-2]].'+$
            'o_fit and n2_Fit are fitted column densities with corresponding radial coordinate, r_fit.'+$
            'sc_ecef is the spacecraft ECEF coordinates.'+$
          'o_ecef,n2_ecef, and fit_ecef are the ecef coordinates for the respective column density profiles in meters.'
         
  save,n_col_o,n_col_n2,o_fit,n2_fit,r_fit,sc_ecef,read_me, $
       filename=fname , o_ecef,n2_ecef,fit_ecef,date_string,time_string



End
