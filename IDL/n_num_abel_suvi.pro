Function n_num_abel_suvi,r_t,n_col_array,  file_path=file_path,r_p=r_p,n_col_ver=n_col_ver;,r_t,n_col
;performs abel transform inversion on input column density
;using exponential smoothing of N_col
;if keyword_set(n_col_ver) eq 1 then stop

r_e=6371.

;make sure r_t and n_col_array are in ascending order
sort_ind=sort(r_t)
r_t=r_t[sort_ind]
n_col_array=n_col_array[sort_ind]

n_col_smooth=n_col_array

r_t_init=r_t
n_col_init=n_col_array

max_meas=max(r_t)

;st_ind=n_elements(n_col_smooth)-50
st_ind=floor(.75*n_elements(n_col_smooth)) ;use upper quartile to fit topside
en_ind=n_elements(n_col_smooth)-1
coefs=poly_fit(r_t[st_ind:en_ind]-r_t[st_ind[0]],alog(n_col_smooth[st_ind:en_ind]),1 )
r_top=linspace(max(r_t),r_e+1000,100)
n_top=exp(coefs[0])*exp(coefs[1]*(r_top-r_t[st_ind[0]] )  )
n_col_smooth=[n_col_smooth,n_top]
r_t=[r_t,r_top]


;window,1
;plot,r_t,n_col_smooth,title='Measured and Exponentially Smoothed Column Densities(---)',charsize=1.5

;fit n_col to local exponential functions

alpha_i=n_col_smooth
beta_i=n_col_smooth

alpha_i[*]=-1.
beta_i[*]=-1.

try_again=1

sm_int=2;5

while try_again eq 1 do begin ;smooth enough to eliminate positive beta

   for ii=sm_int,n_elements(n_col_smooth)-sm_int-1 do begin

      n_col_fit=n_col_smooth[ii-sm_int:ii+sm_int]

      coefs=poly_fit(r_t[ii-sm_int:ii+sm_int]-r_t[ii],alog(n_col_fit) ,1)

      alpha_i[ii]=exp(coefs[0])

      beta_i[ii]=coefs[1]

   end

   ind=where(beta_i gt 0.)

   if ind[0] eq -1 then try_again=0

   sm_int=1.5*sm_int

end


;check the fit

r_test=linspace(min(r_t),max(r_t),10.*n_elements(r_t))
n_est=r_test

for jj=0,n_elements(n_est)-1 do begin

   ind=where(r_t ge  r_test[jj] )

   n_est[jj]=alpha_i[ind[0]]*exp(beta_i[ind[0]]*(r_test[jj]-r_t[ind[0]] ))

end

g_ind=where(alpha_i ne -1.)

;oplot,r_test,n_est,linestyle=2                 


r_t=r_t[g_ind]
alpha_i=alpha_i[g_ind]
beta_i=-1.*beta_i[g_ind] ;formula below assumes beta is positive



;expand coverage to infinity
r=linspace(min(r_t),min(r_t)+2000.,2000)
beta=r
alpha=r
for nn=0,n_elements(r)-1 do begin

ind=where(r_t le r[nn])

beta[nn]=beta_i[max(ind)]

alpha[nn]=alpha_i[max(ind)]

end



r_t0=r_t

;ignore expansion
;r_t=r
;beta_i=beta
;alpha_i=alpha
r=r_t



n_den=r_t

for ll=0,n_elements(n_den)-1 do begin

n_den[ll]=0

for ii=ll,n_elements(n_den)-2 do begin

   n_den[ll]=n_den[ll]+1./(!pi)*alpha_i[ii]*beta_i[ii]/sqrt(r_t[ii]+r_t[ll])* $
            ( (1+0.5*(r_t[ii]-r_t[ll])/(r_t[ii]+r_t[ll])-0.25/(beta_i[ii]*(r_t[ii]+r_t[ll])))* $
             sqrt(!pi/beta_i[ii])* $
             ( sqrt( erf(beta_i[ii]*(r_t[ii+1]-r_t[ll])) ) -sqrt( erf(beta_i[ii]*(r_t[ii]-r_t[ll]) ) ))* $
             exp(beta_i[ii]*(r_t[ii]-r_t[ll]))+ $   
             0.5/(beta_i[ii]*(r_t[ii]+r_t[ll]) )* $
             ( sqrt(r_t[ii+1]-r_t[ll])*exp(-1.*beta_i[ii]*(r_t[ii+1]-r_t[ii]))-sqrt(r_t[ii]-r_t[ll]) ) )/1e5 ;units in cm^-3       


end


end


if keyword_set(n_col_ver) then begin

;compute column density from retrieved density to validate inversion
n_col_verif=r_t

n_col_verif_scaled=r_t

for ll=0,n_elements(r_t)-1 do begin

   n_col_verif[ll]=0.
   n_col_verif_scaled[ll]=0.
   for ii=ll,n_elements(r_t)-2 do begin

      n_col_verif[ll]=n_col_verif[ll]+2.*n_den[ii]*(  sqrt(r_t[ii+1]^2-r_t[ll]^2) - sqrt(r_t[ii]^2-r_t[ll]^2)  )*1e5
      n_col_verif_scaled[ll]=n_col_verif_scaled[ll]+2.*n_den[ii]/.63*(  sqrt(r_t[ii+1]^2-r_t[ll]^2) - sqrt(r_t[ii]^2-r_t[ll]^2)  )*1e5

   end

end

;white_plots

;plot,r_t,n_col_verif,/ylog,yrange=[1e15,1e19],linestyle=1,title='Original Column vs computed from den (...)  and den/.63 (---)',$
;xrange=[6450,6700]
;oplot,r_t_init,n_col_init
;oplot,r_t,n_col_verif_scaled,linestyle=2

;stop

n_col_ver=[transpose(r_t),transpose(n_col_verif)]

end

print,max(r_t0)-r_e

ind=where(r_t le max_meas)

return,[transpose(r_t[ind]),transpose(n_den[ind])]



End
