Function  er_image_to_profile, file,gse_center=gse_center

     r_e=6371. ;km


      d=mrdfits(file,1)

      h_c=d.altitude ;center column altitude
      
      er_image=mrdfits(file,0,h)

      ;get ecef coordinates
     parsed=strsplit(h[-7],' ',/extract)
     ecef_x=double(parsed[1])
     parsed=strsplit(h[-6],' ',/extract)
     ecef_y=double(parsed[1])
     parsed=strsplit(h[-5],' ',/extract)
     ecef_z=double(parsed[1])
     ecef_array=[ecef_x,ecef_y,ecef_z]
     
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

     pos_gse = cxform(ecef_array, 'GEO', 'GSE', time)

     r_center_pix=sqrt(pos_gse[1]^2.+pos_gse[2]^2. )/1e3;km, center pixel altitude

     dx_dn=atan(2.5/3600.*!pi/180.)*abs(pos_gse[0])/1e3;km ;distance in observation plane spanned by each pixel

     pix_val=findgen(n_elements(er_image[*,0]))-n_elements(er_image[*,0])/2+.5 ;pixel count from center column

     pix_val_y=findgen(n_elements(er_image[0,*]))-n_elements(er_image[0,*])/2+.5 ;pixel count from center row


     gse_center=[pos_gse[1],pos_gse[2]]

     r_c=pix_val_y*dx_dn+r_center_pix


;     theta_lo=atan(pix_val*dx_dn/r_c[0])
;     theta_hi=atan(pix_val*dx_dn/r_c[-1])


;     dif_lo=r_c[0]*(1./cos(theta_lo)-1.)

;     dif_hi=r_c[-1]*(1./cos(theta_hi)-1.)


     ;build matrix of pixel altitudes
     
     alt_matrix=[]

     for yy=0,n_elements(r_c)-1 do begin

        theta_yy=atan(pix_val*dx_dn/r_c[yy])

        alt_matrix=[ [alt_matrix],[ r_c[yy]/cos(theta_yy)] ]

     end

    
     ;define altitude grid

     z=findgen(150)*2.+100.+r_e
     er_val=[]

     z_val=[]

;     ind=where(er_image lt 0)
;     er_image[ind]=0

     for zz=1,n_elements(z)-1 do begin


        ind=where( finite(er_image) eq 1 and alt_matrix ge z[zz-1] and alt_matrix lt z[zz]  )

        if ind[0] eq -1 then continue

        z_val=[z_val,mean([z[zz-1],z[zz]] )]


        hist=histogram(er_image[ind],nbins=n_elements(ind)/7.,locations=loc)
        mx_ind=where(hist eq max(hist))



        if n_elements(mx_ind) gt 1 then mode=mean(loc[mx_ind]) else mode=loc[mx_ind]

    
        if n_elements(hist) ge 10 then begin
           result=gaussfit(loc,hist,coefs)

           mode=coefs[1]

        end

;        plot,loc,hist
;        print,mode

        er_val=[er_val,mode]
;        er_val=[er_val,mean(er_image[ind] )]



     end

     return,[[z_val-r_e],[er_val]]


End
