import numpy as np
import tensorflow as tf
import math

class SR1TrustExact:
    
  def __init__(self, loss, var,grad, initialtrustradius = 1.):
    
    self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.atboundary_old = tf.Variable(False, trainable=False)
    self.doiter_old = tf.Variable(False, trainable = False)
    self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    self.grad_old_mag = tf.sqrt(tf.reduce_sum(tf.square(self.grad_old)))
    self.isfirstiter = tf.Variable(True, trainable=False)
    self.UT = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.e = tf.Variable(tf.ones_like(var),trainable=False)
    self.e0 = self.e[0]
    self.doscaling = tf.Variable(False)
    
  def initialize(self, loss, var, grad, B = None):
    alist = []
    alist.append(tf.assign(self.var_old,var))
    alist.append(tf.assign(self.grad_old,grad))
    
    if B is not None:
      e,U = tf.self_adjoint_eig(B)
      UT = tf.transpose(U)
      alist.append(tf.assign(self.e,e))
      alist.append(tf.assign(self.UT,UT))
    return tf.group(alist)
  
  def minimize(self, loss, var, grad = None):
    #TODO, consider replacing gather_nd with gather where appropriate (maybe faster?)
    
    if grad is None:
      grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
    xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    #xtol = 0.
    #eta = 0.
    eta = 0.15
    #eta = 1e-3
    
    #compute ratio of actual reduction in loss function to that
    #predicted from quadratic approximation
    #in order to decide whether to reverse the previous step
    #and whether to enlarge or shrink the trust region
    actual_reduction = self.loss_old - loss
    
    isnull = tf.logical_not(self.doiter_old)
    rho = actual_reduction/self.predicted_reduction
    rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    dgrad = grad - self.grad_old
    dx = var - self.var_old
    dxmag = tf.sqrt(tf.reduce_sum(tf.square(dx)))
  
    trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    #trustradius_out = tf.minimum(trustradius_out,1e10)
    
    #trustradius_out = tf.where(tf.less(rho,0.1),0.5*self.trustradius,
                               #tf.where(tf.less(rho,0.75), self.trustradius,
                               #tf.where(tf.less_equal(dxmag,0.8*self.trustradius), self.trustradius,
                               #2.*self.trustradius)))
                               
    trustradius_out = tf.where(self.doiter_old, trustradius_out, self.trustradius)

    
    #trustradius_out = tf.Print(trustradius_out, [actual_reduction,self.predicted_reduction,rho, trustradius_out], message = "actual_reduction, self.predicted_reduction, rho, trustradius_out: ")
    
    def doSR1Scaling(Bin,yin,dxin):
      s_norm2 = tf.reduce_sum(tf.square(dxin))
      y_norm2 = tf.reduce_sum(tf.square(yin))
      ys = tf.abs(tf.reduce_sum(yin*dxin))
      invalid = tf.equal(ys,0.) | tf.equal(y_norm2, 0.) | tf.equal(s_norm2, 0.)
      scale = tf.where(invalid, tf.ones_like(ys), y_norm2/ys)
      scale = tf.Print(scale,[scale],message = "doing sr1 scaling")
      B = scale*Bin
      return (B,H,tf.constant(False))
    
    #n.b. this has a substantially different form from the usual SR 1 update
    #since we are directly updating the eigenvalue-eigenvector decomposition.
    #The actual hessian approximation is never stored (but memory requirements
    #are similar since the full set of eigenvectors is stored)
    def doSR1Update(ein,UTin,yin,dxin):
      #compute quantities which define the rank 1 update
      #and numerical test to determine whether to perform
      #the update
      y = tf.reshape(yin,[-1,1])
      dx = tf.reshape(dxin,[-1,1])
      ecol = tf.reshape(ein,[-1,1])
      
      UTdx = tf.matmul(UTin, dx)
      UTy = tf.matmul(UTin,y)
      den = tf.matmul(y,dx,transpose_a=True) - tf.matmul(UTdx,ecol*UTdx,transpose_a=True)
      dyBx =  UTy - ecol*UTdx
      dyBxnormsq = tf.reduce_sum(tf.square(dyBx))
      dyBxnorm = tf.sqrt(dyBxnormsq)
      dxnorm = tf.sqrt(tf.reduce_sum(tf.square(dx)))
      dennorm = dxnorm*dyBxnorm
      absden = tf.abs(den)
      dentest = tf.less(absden,1e-8*dennorm) | tf.equal(tf.reshape(absden,[]),0.)
      dentest = tf.reshape(dentest,[])
      dentest = tf.logical_or(dentest,tf.equal(actual_reduction,0.))
      
      
      def doUpdate():
        #compute update in the form
        #B->B + rho zz^T with |z|=1
        z = dyBx/dyBxnorm
        signedrho = dyBxnormsq/den
        signedrho = tf.reshape(signedrho,[])
        #signedrho = tf.Print(signedrho,[signedrho],message="signedrho")
        rho = tf.abs(signedrho)

        flipsign = signedrho < 0.
        
        #in case rho<0, reverse order of eigenvalues and eigenvectors and flip signs
        #to ensure consistent ordering
        #z needs to be reversed as well since it was already computed with the original ordering
        einalt = -tf.reverse(ein,axis=(0,))
        UTinalt = tf.reverse(UTin,axis=(0,))
        zalt = tf.reverse(z,axis=(0,))
        
        estart = tf.where(flipsign,einalt,ein)
        UTstart = tf.where(flipsign,UTinalt,UTin)
        z = tf.where(flipsign,zalt,z)
        
        #estart = tf.Print(estart,[estart],message="estart",summarize=10000)
        
        #deflation in case of repeated eigenvalues
        estartn1 = estart[:-1]
        estart1 = estart[1:]
        ischange = tf.logical_not(tf.equal(estartn1,estart1))
        isfirst = tf.concat([[True],ischange],axis=0)
        islast = tf.concat([ischange,[True]],axis=0)
        islast1 = islast[1:]
        issingle1 = tf.logical_and(ischange,islast1)
        issingle = tf.concat([ischange[0:1], issingle1],axis=0)
        isrep = tf.logical_not(issingle)
        isfirstrep = tf.logical_and(isfirst,isrep)
        islastrep = tf.logical_and(islast,isrep)
        
        firstidxsrep = tf.where(isfirstrep)
        lastidxsrep = tf.where(islastrep)
        rrep = lastidxsrep - firstidxsrep + 1
        rrep = tf.reshape(rrep,[-1])
        
        repidxs = tf.where(isrep)
        firstidxs = tf.where(isfirst)
        lastidxs = tf.where(islast)
        r = lastidxs - firstidxs + 1
        r = tf.reshape(r,[-1])
        isrepunique = r > 1
        uniquerepidxs = tf.where(isrepunique)
        nonlastidxs = tf.where(tf.logical_not(islast))
                
        zflat = tf.reshape(z,[-1])
        
        uniqueidxs = tf.cumsum(tf.cast(islast,tf.int32),exclusive=True)
        xisq2 = tf.segment_sum(tf.square(zflat),uniqueidxs)
        
        xisqrep = tf.gather_nd(xisq2,uniquerepidxs)
        abszrep = tf.sqrt(xisqrep)
        
        #TODO (maybe) skip inflation entirely in case there are no repeating eigenvalues
        
        #loop over sets of repeated eigenvalues in order to perform the necessary
        #transformation of the eigenvectors and rank 1 update
        arrsize = tf.shape(firstidxsrep)[0]
        arr0 = tf.TensorArray(var.dtype,size=arrsize,infer_shape=False,element_shape=[None,var.shape[0]])
        deflate_var_list = [arr0, tf.constant(0,dtype=tf.int32)]
        def deflate_cond(arr,j):
          return j<arrsize
        def deflate_body(arr,j):
          size = rrep[j]
          startidx = tf.reshape(firstidxsrep[j],[])
          endidx = startidx + size
          zsub = zflat[startidx:endidx]
          UTsub = UTstart[startidx:endidx]
          magzsub = abszrep[j]
          en = tf.one_hot(size-1,depth=tf.cast(size,tf.int32),dtype=zsub.dtype)
          #this is the vector which implicitly defines the Householder transformation matrix
          v = zsub/magzsub + en
          v = v/tf.sqrt(tf.reduce_sum(tf.square(v)))
          v = tf.reshape(v,[-1,1])
          #protection for v~=0 case (when zsub~=-en), then no transformation is needed
          nullv = tf.reduce_all(tf.equal(tf.sign(zsub),-en))
          v = tf.where(nullv,tf.zeros_like(v),v)
          UTbarsub = UTsub - 2.*tf.matmul(v,tf.matmul(v,UTsub,transpose_a=True))
          arr = arr.write(j,UTbarsub)
          return (arr, j+1)
        
        UTbararr,j = tf.while_loop(deflate_cond,deflate_body,deflate_var_list, parallel_iterations=64, back_prop=False)
        UTbarrep = UTbararr.concat()
        
        #reassemble transformed eigenvectors and update vector
        #now z=0 for repeated eigenvectors except for the last instance
        UTbar = tf.where(issingle, UTstart, tf.scatter_nd(repidxs,UTbarrep, shape=UTstart.shape))
        zbar = tf.where(issingle, zflat, tf.scatter_nd(lastidxsrep,-abszrep, shape=zflat.shape))
                
        #construct deflated system consisting of unique eigenvalues only
        UT1 = tf.gather_nd(UTbar,nonlastidxs)     
        UT2 = tf.gather_nd(UTbar,lastidxs)
        e1 = tf.gather_nd(estart,nonlastidxs)
        d = tf.gather_nd(estart,lastidxs)
        z2 = tf.gather_nd(zbar,lastidxs)
        
        #TODO, check if this reshape is really needed)
        xisq = tf.reshape(xisq2,[-1])
        
        #compute quantities needed for eigenvalue update
        dnorm = d/rho
        dnormn1 = dnorm[:-1]
        dnorm1 = dnorm[1:]
        dn1 = d[:-1]
        d1 = d[1:]
        deltan1 = dnorm1 - dnormn1
        delta = tf.concat([deltan1,[1.]],axis=0)
        rdeltan1 = rho/(d1-dn1)
        rdelta = tf.concat([rdeltan1,[1.]],axis=0)
        rdelta2 = tf.square(rdelta)
        xisqn1 = tf.reshape(xisq[:-1],[-1])
        xisq1n1 = tf.reshape(xisq[1:],[-1])
        xisq1 = tf.concat([xisq1n1, [0.]],axis=0)
        
        dnormi = tf.reshape(dnorm,[-1,1])
        dnormj = tf.reshape(dnorm,[1,-1])
        deltam = dnormj - dnormi
        deltam2 = tf.concat([deltam[1:],deltam[-1:]-tf.ones_like(dnormj)],axis=0)
                        
        t0 = tf.zeros_like(d)

        nupper = tf.minimum(1,tf.shape(d)[0])
        deltambool = tf.ones_like(deltam,dtype=tf.bool)
        deltamones = tf.ones_like(deltam)
        deltamask = tf.matrix_band_part(deltambool,tf.zeros_like(nupper),nupper)
        
        deltamaskdiag = tf.matrix_band_part(deltambool,0,0)
        deltamasklow  = tf.matrix_band_part(deltambool,-1,0) & tf.logical_not(deltamaskdiag)
        deltamaskhigh  = tf.matrix_band_part(deltambool,0,-1)
                
        unconverged0 = tf.ones_like(d,dtype=tf.bool)
                  
        loop_vars = [t0,unconverged0,tf.constant(0),t0]
        def cond(yd,unconverged,j,phi):
          return tf.reduce_any(unconverged) & (j<50)
        
        #solution to eigenvalue update in terms of t = (dout - din)/rho
        def body(t,unconverged,j,phi):
          dt = delta - t
          
          t2 = tf.square(t)
          t3 = t*t2
          
          dt2 = tf.square(dt)
          dt3 = dt*dt2
          
          ti = tf.reshape(t,[-1,1])
          frden = tf.reciprocal(deltam-ti)
          #exclude j=i,i+1 terms
          frden = tf.where(deltamask,tf.zeros_like(frden),frden)
          issingular = tf.reduce_any(tf.is_inf(frden),axis=-1)
          xisqj = tf.reshape(xisq,[1,-1])
          s0arg = xisqj*frden
          s1arg = s0arg*frden
          s2arg = s1arg*frden
          
          s0 = tf.reduce_sum(s0arg, axis=-1)
          s1 = tf.reduce_sum(s1arg, axis=-1)
          s2 = tf.reduce_sum(s2arg, axis=-1)
          
          #function value is not actually used, but computed
          #for diagnostic purposes only
          phi = t*dt*(1.+s0) - dt*xisq + t*xisq1
          phi = tf.where(tf.is_nan(phi),tf.zeros_like(phi),phi)
          magw = tf.sqrt(tf.reduce_sum(tf.square(phi)))

          cg = (dt3*s1 + t*dt3*s2)*rdelta + xisq1
          bg = (-t3*s1 + t3*dt*s2)*rdelta - xisq
          ag = 1. + s0 + (2.*t - delta)*s1 - t*dt*s2
          
          a = ag
          b = -delta*ag + bg - cg
          c = t3*s1 - t3*dt*s2 + delta*xisq
          
          told = t
          
          #use three different forms of the quadratic formula depending on the sign of b
          #in order to avoid cancellations/roundoff or nan
          sarg = tf.square(b) - 4.*a*c
          sarg = tf.maximum(sarg,0.)
          s = tf.sqrt(sarg)
          tnom = -0.5*(b+s)/a
          talt = -2.*c/(b-s)
          #with protection for roundoff error which could make -c/a negative
          s2arg = -c/a
          s2arg = tf.maximum(s2arg,0.)
          ts = tf.sqrt(s2arg)
          
          signb = tf.sign(b)
          t = tf.where(tf.equal(signb,0), ts,tf.where(signb>0, tnom, talt))
          
          #protection for singular case
          t = tf.where(issingular, tf.zeros_like(t), t)
          
          #roundoff errors could produce solutions out of bounds for t->delta
          #but these will be discarded later anyways in favor of the alternate solution
          t = tf.maximum(t,0.)
          t = tf.minimum(t,delta)
          
          #when individual eigenvalues have converged we mark them as such
          #but simply keep iterating on the full vector, since any efficiency
          #gains from partially stopping and chopping up the vector would likely incur
          #more overhead, especially on GPU
          tadvancing = t > told
          #if t>0.5*delta we won't use this solution anyways, so we don't care if its converged or not
          #(but leave an extra margin here to avoid any possible numerical issues)
          tunsaturated = t < 0.6*delta
          unconverged = unconverged & tadvancing & tunsaturated
                                 
          #t = tf.Print(t,[magw],message="magw")
          
          return (t,unconverged,j+1,phi)
          
          
        t,unconverged,j,phi = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
        
        tassert = tf.Assert(tf.reduce_all((t>=0.) & (t<=delta)),[t],summarize=10000)        
        with tf.control_dependencies([tassert]):
          t = tf.identity(t)
        
        #solution to eigenvalue update in terms of delta-t
        def body2(t,unconverged,j,phi):
          dt = delta - t
          
          t2 = tf.square(t)
          t3 = t*t2
          
          dt2 = tf.square(dt)
          dt3 = dt*dt2
          
          ti = tf.reshape(t,[-1,1])
          frden = tf.reciprocal(deltam2+ti)
          #exclude j=i,i+1 terms
          frden = tf.where(deltamask,tf.zeros_like(frden),frden)
          issingular = tf.reduce_any(tf.is_inf(frden),axis=-1)
          xisqj = tf.reshape(xisq,[1,-1])
          s0arg = xisqj*frden
          s1arg = s0arg*frden
          s2arg = s1arg*frden
          
          s0 = tf.reduce_sum(s0arg, axis=-1)
          s1 = tf.reduce_sum(s1arg, axis=-1)
          s2 = tf.reduce_sum(s2arg, axis=-1)
          
          phi = t*dt*(1.+s0) - t*xisq + dt*xisq1
          phi = tf.where(tf.is_nan(phi),tf.zeros_like(phi),phi)
          #phi = 1.+s0 - xisq/dt + xisq1/t
          magw = tf.sqrt(tf.reduce_sum(tf.square(phi)))

          cg = (-dt3*s1 + t*dt3*s2)*rdelta - xisq
          bg = (t3*s1 + t3*dt*s2)*rdelta + xisq1
          ag = 1. + s0 - (2.*t - delta)*s1 - t*dt*s2
          
          a = ag
          b = -delta*ag + bg - cg
          c = -t3*s1 - t3*dt*s2 - delta*xisq1
          
          told = t
          
          #use two different forms of the quadratic formula depending on the sign of b
          #in order to avoid cancellations/roundoff
          sarg = tf.square(b) - 4.*a*c
          sarg = tf.maximum(sarg,0.)
          s = tf.sqrt(sarg)
          tnom = -0.5*(b-s)/a
          talt = -2.*c/(b+s)
          #with protection for roundoff error which could make -c/a negative
          s2arg = -c/a
          s2arg = tf.maximum(s2arg,0.)
          ts = tf.sqrt(s2arg)
          
          signb = tf.sign(b)
          t = tf.where(tf.equal(signb,0), ts,tf.where(signb>0, talt, tnom))
          
          #protection for singular case
          t = tf.where(issingular, tf.zeros_like(t), t)
          
          #roundoff errors could produce solutions out of bounds for t->delta
          #but these will be discarded later anyways in favor of the alternate solution
          t = tf.maximum(t,0.)
          t = tf.minimum(t,delta)
          
          #when individual eigenvalues have converged we mark them as such
          #but simply keep iterating on the full vector, since any efficiency
          #gains from partially stopping and chopping up the vector would likely incur
          #more overhead, especially on GPU
          tadvancing = t > told
          #if t>0.5*delta we won't use this solution anyways, so we don't care if its converged or not
          #(but leave an extra margin here to avoid any possible numerical issues)
          tunsaturated = t < 0.6*delta
          #tunsaturated = tadvancing
          unconverged = unconverged & tadvancing & tunsaturated

          #t = tf.Print(t,[magw],message="magw2")
          
          return (t,unconverged,j+1,phi)
        
        dt,unconverged2,j2,phi2 = tf.while_loop(cond, body2, loop_vars, parallel_iterations=1, back_prop=False)
        
        t2assert = tf.Assert(tf.reduce_all((dt>=0.) & (dt<=delta)),[dt],summarize=10000)        
        with tf.control_dependencies([t2assert]):
          dt = tf.identity(dt)
                
        d1 = tf.concat([d[1:],d[-1:]+rho],axis=0)
        dout = d + rho*t
        dout2 = d1 - rho*dt
        #choose solution with higher numerical precision
        tswitch = t <= dt
        dout = tf.where(tswitch,dout, dout2)
        phiout = tf.where(tswitch,phi,phi2)
        magphi = tf.reduce_sum(tf.square(phiout))
        
        #recompute z as in https://doi.org/10.1137/S089547989223924X to maintain orthogonality of eigenvectors
        deltam1 = deltam[:,1:]
        
        deltam1num = tf.concat([deltam1,1.+deltam[:,-1:]],axis=-1)
        deltam1den = tf.concat([deltam1,tf.ones_like(deltam[:,-1:])],axis=-1)
        
        prodmnum = deltam + tf.reshape(t,[1,-1])
        prodmnum2 = deltam1num - tf.reshape(dt,[1,-1])
        prodmnumswitch = tf.reshape(tswitch,[1,-1]) | tf.zeros_like(prodmnum, dtype=tf.bool)
        prodmnum = tf.where(prodmnumswitch,prodmnum,prodmnum2)
        
        prodmdenlow = tf.where(deltamasklow,deltam,tf.zeros_like(deltam))
        prodmdenhigh = tf.where(deltamaskhigh,deltam1den,tf.zeros_like(deltam1den))
        
        prodmden = prodmdenlow + prodmdenhigh
        
        nullden = tf.equal(prodmden,0.)
        prodr = tf.where(nullden,tf.ones_like(prodmnum),prodmnum*tf.reciprocal(tf.where(nullden,tf.ones_like(prodmden),prodmden)))
        prod = tf.reduce_prod(prodr,axis=-1)
        absztilde = tf.sqrt(prod)
        ztilde = tf.where(tf.greater_equal(z2,0.),absztilde,-absztilde)
        
        #now compute eigenvectors, with rows of this matrix constructed
        #from the solution with the higher numerical precision
        ti = tf.reshape(t,[-1,1])
        dti = tf.reshape(dt,[-1,1])
        tswitchi = tf.reshape(tswitch,[-1,1])
        D = deltam - ti
        D2 = deltam2 + dti
        Dswitch = tf.reshape(tswitch,[-1,1]) | tf.zeros_like(D, dtype=tf.bool)

        D = tf.where(Dswitch,D,D2)
        Dinv = tf.reciprocal(tf.where(tf.equal(D,0.),tf.ones_like(D),D))

        Dinvz = Dinv*tf.reshape(ztilde,[1,-1])
        Dinvzmag = tf.sqrt(tf.reduce_sum(tf.square(Dinvz),axis=-1,keepdims=True))
        Dinvz = Dinvz/Dinvzmag
        
        #n.b. this is the most expensive operation (matrix-matrix multiplication to compute the updated eigenvectors)
        UT2out = tf.matmul(Dinvz,UT2)
        
        #protections for t=0 or t=delta cases
        #if t=0 the eigenvector is unchanged
        #if t=delta then the i+1st eigenvector is shifted
        #to the ith position
        UT21 = tf.concat([UT2[1:],UT2[-1:]],axis=0)      
        tnull = tf.equal(ti,0.) & tswitchi
        dtnull = tf.equal(dti,0.) & tf.logical_not(tswitchi)
        UT2false = tf.zeros_like(UT2,dtype=tf.bool)
        tnullm = tf.logical_or(UT2false,tnull)
        dtnullm = tf.logical_or(UT2false,dtnull)
        
        UT2out = tf.where(dtnullm,UT21,UT2out)
        UT2out = tf.where(tnullm,UT2,UT2out)
                        
        #now put everything back together
        #eigenvalues are still guaranteed to be sorted
        eout = tf.scatter_nd(lastidxs,dout,estart.shape) + tf.scatter_nd(nonlastidxs,e1,estart.shape)
        UTout = tf.scatter_nd(lastidxs,UT2out,UTstart.shape) + tf.scatter_nd(nonlastidxs,UT1,UTstart.shape)
        
        #restore correct order and signs if necessary
        eoutalt = -tf.reverse(eout,axis=(0,))
        UToutalt = tf.reverse(UTout,axis=(0,))
        
        eout = tf.where(flipsign,eoutalt,eout)
        UTout = tf.where(flipsign,UToutalt,UTout)
                
        return (eout,UTout)
      
      e,UT = tf.cond(dentest, lambda: (ein,UTin), doUpdate)
      
      return (e,UT)
    
    esec = self.e
    UTsec = self.UT
    
    doscaling = tf.constant(False)
    #B,H,doscaling = tf.cond(self.doscaling & self.doiter_old, lambda: doSR1Scaling(B,H,dgrad,dx), lambda: (B,H,self.doscaling))
    esec,UTsec = tf.cond(self.doiter_old, lambda: doSR1Update(esec,UTsec,dgrad,dx), lambda: (esec,UTsec))  
    
    isconvergedxtol = trustradius_out < xtol
    isconvergededmtol = self.predicted_reduction <= 0.
    
    isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    doiter = tf.logical_and(tf.greater(rho,eta),tf.logical_not(isconverged))
    
    
    def build_sol():

      lam = esec
      UT = UTsec
      
      gradcol = tf.reshape(grad,[-1,1])
      
      #projection of gradient onto eigenvectors
      a = tf.matmul(UT, gradcol)
      a = tf.reshape(a,[-1])
      
      amagsq = tf.reduce_sum(tf.square(a))
      gmagsq = tf.reduce_sum(tf.square(grad))
      
      asq = tf.square(a)
      
      #deal with null gradient components and repeated eigenvectors
      lamn1 = lam[:-1]
      lam1 = lam[1:]
      ischange = tf.logical_not(tf.equal(lamn1,lam1))
      islast = tf.concat([ischange,[True]],axis=0)
      lastidxs = tf.where(islast)
      uniqueidx = tf.cumsum(tf.cast(islast,tf.int32),exclusive=True)
      uniqueasq = tf.segment_sum(asq,uniqueidx)
      uniquelam = tf.gather_nd(lam,lastidxs)
      
      abarindices = tf.where(uniqueasq)
      abarsq = tf.gather_nd(uniqueasq,abarindices)
      lambar = tf.gather_nd(uniquelam,abarindices)
      
      #abar = tf.sqrt(abarsq)
      abarmag = tf.sqrt(abarsq)
      
      e0 = lam[0]
      sigma0 = tf.maximum(-e0,tf.zeros([],dtype=var.dtype))
      
      def phif(s):        
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        pmag = tf.sqrt(pmagsq)
        phipartial = tf.reciprocal(pmag)
        singular = tf.reduce_any(tf.equal(-s,lambar))
        phipartial = tf.where(singular, tf.zeros_like(phipartial), phipartial)
        phi = phipartial - tf.reciprocal(trustradius_out)
        return phi
      
      def phiphiprime(s):
        phi = phif(s)
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))        
        phiprime = tf.pow(pmagsq,-1.5)*tf.reduce_sum(abarsq/tf.pow(lambar+s,3))
        return (phi, phiprime)
        
      
      #TODO, add handling of additional cases here (singular and "hard" cases)
      
      phisigma0 = phif(sigma0)
      usesolu = tf.logical_and(e0>0. , phisigma0 >= 0.)
      
      def sigma():
        #tol = 1e-10
        maxiter = 50

        sigmainit = tf.reduce_max(tf.abs(a)/trustradius_out - lam)
        sigmainit = tf.maximum(sigmainit,tf.zeros_like(sigmainit))
        phiinit,phiprimeinit = phiphiprime(sigmainit)
                
        loop_vars = [sigmainit, phiinit,phiprimeinit, tf.constant(True), tf.zeros([],dtype=tf.int32)]
        
        def cond(sigma,phi,phiprime,unconverged,j):
          return (unconverged) & (j<maxiter)
        
        def body(sigma,phi,phiprime,unconverged,j):   
          sigmaout = sigma - phi/phiprime
          phiout, phiprimeout = phiphiprime(sigmaout)
          unconverged = (phiout > phi) & (phiout < 0.)
          #phiout = tf.Print(phiout,[phiout],message="phiout")
          return (sigmaout,phiout,phiprimeout,unconverged,j+1)
          
        sigmaiter, phiiter,phiprimeiter,unconverged,jiter = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
        #sigmaiter = tf.Print(sigmaiter,[phiiter],message="phiiter")
        return sigmaiter
      
      #sigma=0 corresponds to the unconstrained solution on the interior of the trust region
      sigma = tf.cond(usesolu, lambda: tf.zeros([],dtype=var.dtype), sigma)

      #solution can be computed directly from eigenvalues and eigenvectors
      coeffs = -a/(lam+sigma)
      coeffs = tf.reshape(coeffs,[1,-1])
      #p = tf.reduce_sum(coeffs*U, axis=-1)
      p = tf.matmul(UT,tf.reshape(coeffs,[-1,1]),transpose_a=True)
      p = tf.reshape(p,[-1])

      Umag = tf.sqrt(tf.reduce_sum(tf.square(UT),axis=1))
      coeffsmag = tf.sqrt(tf.reduce_sum(tf.square(coeffs)))
      pmag = tf.sqrt(tf.reduce_sum(tf.square(p)))
      #the equivalence of |p| and |coeffs| is a partial test of the orthonormality of the eigenvectors
      #which could be degraded in case of excessive loss of numerical precision
      #p = tf.Print(p,[pmag,coeffsmag,sigma],message="pmag,coeffsmag,sigma")

      #predicted reduction also computed directly from eigenvalues and eigenvectors
      predicted_reduction_out = -(tf.reduce_sum(a*coeffs) + 0.5*tf.reduce_sum(lam*tf.square(coeffs)))
      
      return [var+p, predicted_reduction_out, tf.logical_not(usesolu), grad]

    #doiter = tf.Print(doiter,[doiter],message="doiter")
    loopout = tf.cond(doiter, lambda: build_sol(), lambda: [self.var_old+0., tf.zeros_like(loss),tf.constant(False),self.grad_old])
    var_out, predicted_reduction_out, atboundary_out, grad_out = loopout
    
    #assign updated values to stored variables, taking care to define dependencies such that things are executed
    #in the correct order
    alist = []
    with tf.control_dependencies(loopout):
      oldvarassign = tf.assign(self.var_old,var)
      alist.append(oldvarassign)
      alist.append(tf.assign(self.loss_old,loss))
      alist.append(tf.assign(self.doiter_old, doiter))
      alist.append(tf.assign(self.doscaling,doscaling))
      alist.append(tf.assign(self.grad_old,grad_out))
      alist.append(tf.assign(self.predicted_reduction,predicted_reduction_out))
      alist.append(tf.assign(self.atboundary_old, atboundary_out))
      alist.append(tf.assign(self.trustradius, trustradius_out))
      alist.append(tf.assign(self.isfirstiter,False)) 
      alist.append(tf.assign(self.e,esec)) 
      alist.append(tf.assign(self.UT,UTsec)) 
       
    clist = []
    clist.extend(loopout)
    clist.append(oldvarassign)
    with tf.control_dependencies(clist):
      varassign = tf.assign(var, var_out)
      
      alist.append(varassign)
      return [isconverged,tf.group(alist)]


class SR1TrustOBS:
    
  def __init__(self, loss, var,grad, initialtrustradius = 1.):
    
    self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.atboundary_old = tf.Variable(False, trainable=False)
    self.doiter_old = tf.Variable(False, trainable = False)
    self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    self.isfirstiter = tf.Variable(True, trainable=False)
    self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.doscaling = tf.Variable(False)
    
  def initialize(self, loss, var, grad, B = None, H = None):
    alist = []
    alist.append(tf.assign(self.var_old,var))
    alist.append(tf.assign(self.grad_old,grad))
    
    if B is not None and H is not None:
      alist.append(tf.assign(self.B,B))
      alist.append(tf.assign(self.H,H))
    return tf.group(alist)
  
    
  #def initialize(self, loss, var, k=7, initialtrustradius = 1.):
    #self.k = k
    
    #self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    #self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    #self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    #self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    #self.atboundary_old = tf.Variable(False, trainable=False)
    #self.doiter_old = tf.Variable(True, trainable = False)
    #self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    #self.isfirstiter = tf.Variable(True, trainable=False)
    ##self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    ##self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    #self.ST = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.YT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.psi = tf.Variable(tf.zeros([var.shape[0],0],dtype=var.dtype),trainable = False)
    #self.M = tf.Variable(tf.zeros([0,0],dtype=var.dtype),trainable = False)
    ##self.gamma = tf.constant(tf.ones([1],dtype=var.dtype), trainable = False)
    #self.gamma = tf.ones([1],dtype=var.dtype)
    ##self.doscaling = tf.Variable(True)
    #self.updateidx = tf.Variable(tf.zeros([1],dtype=tf.int32),trainable = False)
    #self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
        
    #alist = []
    #alist.append(tf.assign(self.trustradius,initialtrustradius))
    #alist.append(tf.assign(self.loss_old, loss))
    #alist.append(tf.assign(self.predicted_reduction, 0.))
    #alist.append(tf.assign(self.var_old, var))
    #alist.append(tf.assign(self.atboundary_old,False))
    #alist.append(tf.assign(self.doiter_old,False))
    #alist.append(tf.assign(self.isfirstiter,True))
    #alist.append(tf.assign(self.ST,tf.zeros_like(self.ST))
    #alist.append(tf.assign(self.YT,tf.zeros_like(self.YT))
    ##alist.append(tf.assign(self.doscaling,True))
    #alist.append(tf.assign(self.grad_old,self.grad))
    
    ##if doScaling
    

    #return tf.group(alist)

  
  def minimize(self, loss, var, grad = None):
    
    if grad is None:
      grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
    
    xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    #edmtol = math.sqrt(xtol)
    #edmtol = xtol
    #edmtol = 1e-8
    #edmtol = 0.
    #eta = 0.
    eta = 0.15
    
          
    actual_reduction = self.loss_old - loss
    
    #actual_reduction = tf.Print(actual_reduction,[self.loss_old, loss, actual_reduction])
    isnull = tf.logical_not(self.doiter_old)
    rho = actual_reduction/self.predicted_reduction
    rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    dgrad = grad - self.grad_old
    dx = var - self.var_old
    dxmag = tf.sqrt(tf.reduce_sum(tf.square(dx)))
  
    trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    #trustradius_out = tf.minimum(trustradius_out,1e10)
    
    #trustradius_out = tf.where(tf.less(rho,0.1),0.5*self.trustradius,
                               #tf.where(tf.less(rho,0.75), self.trustradius,
                               #tf.where(tf.less_equal(dxmag,0.8*self.trustradius), self.trustradius,
                               #2.*self.trustradius)))
                               
    trustradius_out = tf.where(self.doiter_old, trustradius_out, self.trustradius)

    
    trustradius_out = tf.Print(trustradius_out, [actual_reduction,self.predicted_reduction,rho, trustradius_out], message = "actual_reduction, self.predicted_reduction, rho, trustradius_out: ")
    
    #def hesspexact(v):
      #return tf.gradients(self.grad*tf.stop_gradient(v),var, gate_gradients=True)[0]    
    
    #def hesspapprox(B,v):
      #return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])    
    
    #def Bv(gamma,psi,M,vcol):
      #return gamma*vcol + tf.matmul(psi,tf.matmul(M,tf.matmul(psi,vcol,transpose_a=True)))
    
    #def Bvflat(gamma,psi,M,v):
      #vcol = tf.reshape(v,[-1,1])
      #return tf.reshape(Bv(gamma,psi,M,vcol),[-1])
      
    def Bv(gamma,psi,MpsiT,vcol):
      return gamma*vcol + tf.matmul(psi,tf.matmul(MpsiT,vcol))
    
    def Bvflat(gamma,psi,MpsiT,v):
      vcol = tf.reshape(v,[-1,1])
      return tf.reshape(Bv(gamma,psi,MpsiT,vcol),[-1])
    
    def hesspexact(v):
      return tf.gradients(self.grad*tf.stop_gradient(v),var, gate_gradients=True)[0]    
    
    def hesspapprox(B,v):
      return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])    
    
    def doSR1Scaling(Bin,Hin,yin,dxin):
      s_norm2 = tf.reduce_sum(tf.square(dxin))
      y_norm2 = tf.reduce_sum(tf.square(yin))
      ys = tf.abs(tf.reduce_sum(yin*dxin))
      invalid = tf.equal(ys,0.) | tf.equal(y_norm2, 0.) | tf.equal(s_norm2, 0.)
      scale = tf.where(invalid, tf.ones_like(ys), y_norm2/ys)
      scale = tf.Print(scale,[scale],message = "doing sr1 scaling")
      B = scale*Bin
      H = Hin/scale
      return (B,H,tf.constant(False))
    
    def doSR1Update(Bin,Hin,yin,dxin):
      y = tf.reshape(yin,[-1,1])
      dx = tf.reshape(dxin,[-1,1])
      Bx = tf.matmul(Bin,dx)
      dyBx = y - Bx
      den = tf.matmul(dyBx,dx,transpose_a=True)
      deltaB = tf.matmul(dyBx,dyBx,transpose_b=True)/den
      dennorm = tf.sqrt(tf.reduce_sum(tf.square(dx)))*tf.sqrt(tf.reduce_sum(tf.square(dyBx)))
      dentest = tf.less(tf.abs(den),1e-8*dennorm)
      dentest = tf.reshape(dentest,[])
      dentest = tf.logical_or(dentest,tf.equal(actual_reduction,0.))
      deltaB = tf.where(dentest,tf.zeros_like(deltaB),deltaB)
      #deltaB = tf.where(self.doiter_old, deltaB, tf.zeros_like(deltaB))
      
      Hy = tf.matmul(Hin,y)
      dxHy = dx - Hy
      deltaH = tf.matmul(dxHy,dxHy,transpose_b=True)/tf.matmul(dxHy,y,transpose_a=True)
      deltaH = tf.where(dentest,tf.zeros_like(deltaH),deltaH)
      #deltaH = tf.where(self.doiter_old, deltaH, tf.zeros_like(deltaH))
      
      B = Bin + deltaB
      H = Hin + deltaH
      return (B,H)
    
    #grad = self.grad
    B = self.B
    H = self.H
    
    #dgrad = grad - self.grad_old
    #dx = var - self.var_old
    doscaling = tf.constant(False)
    #B,H,doscaling = tf.cond(self.doscaling & self.doiter_old, lambda: doSR1Scaling(B,H,dgrad,dx), lambda: (B,H,self.doscaling))
    B,H = tf.cond(self.doiter_old, lambda: doSR1Update(B,H,dgrad,dx), lambda: (B,H))  
    
  
    
    #psi = tf.Print(psi,[psi],message="psi: ")
    #M = tf.Print(M,[M],message="M: ")
    
    isconvergedxtol = trustradius_out < xtol
    #isconvergededmtol = tf.logical_not(self.isfirstiter) & (self.predicted_reduction <= 0.)
    isconvergededmtol = self.predicted_reduction <= 0.
    
    isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    doiter = tf.logical_and(tf.greater(rho,eta),tf.logical_not(isconverged))
    
    #doiter = tf.Print(doiter, [doiter, isconvergedxtol, isconvergededmtol,isconverged,trustradius_out])
    
    def build_sol():

      lam,U = tf.self_adjoint_eig(B) #TODO: check if what is returned here should actually be UT in the paper
      #U = tf.transpose(U)

      
      #R = tf.Print(R,[detR],message = "detR")

      
      #Rinverse = tf.matrix_inverse(R)
      
      gradcol = tf.reshape(grad,[-1,1])
      
      a = tf.matmul(U, gradcol,transpose_a=True)
      a = tf.reshape(a,[-1])
      
      amagsq = tf.reduce_sum(tf.square(a))
      gmagsq = tf.reduce_sum(tf.square(grad))
      
      a = tf.Print(a,[amagsq,gmagsq],message = "amagsq,gmagsq")
      
      #a = tf.matmul(U, gradcol,transpose_a=False)
      asq = tf.square(a)
      

      abarindices = tf.where(asq)
      abarsq = tf.gather(asq,abarindices)
      lambar = tf.gather(lam,abarindices)

      abarsq = tf.reshape(abarsq,[-1])
      lambar = tf.reshape(lambar, [-1])
      
      lambar, abarindicesu = tf.unique(lambar)
      abarsq = tf.unsorted_segment_sum(abarsq,abarindicesu,tf.shape(lambar)[0])
      
      abar = tf.sqrt(abarsq)
      
      #abarsq = tf.square(abar)


      #nv = tf.shape(ST)[0]
      #I = tf.eye(int(var.shape[0]),dtype=var.dtype)
      #B = gamma*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #B = tf.Print(B, [B],message="B: ", summarize=1000)
      #efull = tf.self_adjoint_eigvals(B)
      #lam = efull[:1+nv]
      
      e0 = lam[0]
      sigma0 = tf.maximum(-e0,tf.zeros([],dtype=var.dtype))
      
      
      #lambar, lamidxs = tf.unique(lam)
      #abarsq = tf.segment_sum(asq,lamidxs)
      
      abarsq = tf.Print(abarsq, [a, abar, lam, lambar], message = "a,abar,lam,lambar")
      
      
      def phif(s):        
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        pmag = tf.sqrt(pmagsq)
        phipartial = tf.reciprocal(pmag)
        singular = tf.reduce_any(tf.equal(-s,lambar))
        #singular = tf.logical_or(singular, tf.is_nan(phipartial))
        #singular = tf.logical_or(singular, tf.is_inf(phipartial))
        phipartial = tf.where(singular, tf.zeros_like(phipartial), phipartial)
        phi = phipartial - tf.reciprocal(trustradius_out)
        return phi
      
      def phiphiprime(s):
        phi = phif(s)
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        phiprime = tf.pow(pmagsq,-1.5)*tf.reduce_sum(abarsq/tf.pow(lambar+s,3))
        return (phi, phiprime)
        
      
      phisigma0 = phif(sigma0)
      #usesolu = e0>0. & phisigma0 >= 0.
      usesolu = tf.logical_and(e0>0. , phisigma0 >= 0.)
      usesolu = tf.Print(usesolu,[sigma0,phisigma0,usesolu], message = "sigma0, phisigma0,usesolu: ")

      def solu():
        return -tf.matmul(H,gradcol)
      
      def sol():
        tol = 1e-8
        maxiter = 50

        sigmainit = tf.reduce_max(tf.abs(a)/trustradius_out - lam)
        sigmainit = tf.maximum(sigmainit,tf.zeros_like(sigmainit))
        phiinit = phif(sigmainit)
        
        sigmainit = tf.Print(sigmainit,[sigmainit,phiinit],message = "sigmainit, phinit: ")

        
        loop_vars = [sigmainit, phiinit, tf.zeros([],dtype=tf.int32)]
        
        def cond(sigma,phi,j):
          #phi = tf.Print(phi,[phi],message = "checking phi in cond()")
          return tf.logical_and(phi < -tol, j<maxiter)
          #return tf.logical_and(tf.abs(phi) > tol, j<maxiter)
        
        def body(sigma,phi,j):   
          #sigma = tf.Print(sigma, [sigma, phi], message = "sigmain, phiin: ")
          phiout, phiprimeout = phiphiprime(sigma)
          sigmaout = sigma - phiout/phiprimeout
          sigmaout = tf.Print(sigmaout, [sigmaout,phiout, phiprimeout], message = "sigmaout, phiout, phiprimeout: ")
          return (sigmaout,phiout,j+1)
          
        sigmaiter, phiiter, jiter = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
        #sigmaiter = tf.Print(sigmaiter,[sigmaiter,phiiter],message = "sigmaiter,phiiter")
        
        coeffs = -a/(lam+sigmaiter)
        coeffs = tf.reshape(coeffs,[1,-1])
        p = tf.reduce_sum(coeffs*U, axis=-1)
        
        return p
      
      p = tf.cond(usesolu, solu, sol)
      p = tf.reshape(p,[-1])
      
      magp = tf.sqrt(tf.reduce_sum(tf.square(p)))
      p = tf.Print(p,[magp],message = "magp")

      #e0val = efull[0]
      #e0val = e0
      
      #Bfull = tau*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #pfull = -tf.matrix_solve(Bfull,tf.reshape(grad,[-1,1]))
      #pfull = tf.reshape(p,[-1])
      #p = pfull

      #p  = tf.Print(p,[e0,e0val,sigma0,sigma,tau], message = "e0, e0val, sigma0, sigma, tau")
      #p  = tf.Print(p,[lam,efull], message = "lam, efull")

      predicted_reduction_out = -(tf.reduce_sum(grad*p) + 0.5*tf.reduce_sum(tf.reshape(tf.matmul(B,tf.reshape(p,[-1,1])),[-1])*p) )
      
      return [var+p, predicted_reduction_out, tf.logical_not(usesolu), grad]

    loopout = tf.cond(doiter, lambda: build_sol(), lambda: [self.var_old+0., tf.zeros_like(loss),tf.constant(False),self.grad_old])
    var_out, predicted_reduction_out, atboundary_out, grad_out = loopout
        
    #var_out = tf.Print(var_out,[],message="var_out")
    #loopout[0] = var_out
    
    alist = []
    
    with tf.control_dependencies(loopout):
      oldvarassign = tf.assign(self.var_old,var)
      alist.append(oldvarassign)
      alist.append(tf.assign(self.loss_old,loss))
      alist.append(tf.assign(self.doiter_old, doiter))
      alist.append(tf.assign(self.B,B))
      alist.append(tf.assign(self.H,H))
      alist.append(tf.assign(self.doscaling,doscaling))
      alist.append(tf.assign(self.grad_old,grad_out))
      alist.append(tf.assign(self.predicted_reduction,predicted_reduction_out))
      alist.append(tf.assign(self.atboundary_old, atboundary_out))
      alist.append(tf.assign(self.trustradius, trustradius_out))
      alist.append(tf.assign(self.isfirstiter,False)) 
       
    clist = []
    clist.extend(loopout)
    clist.append(oldvarassign)
    with tf.control_dependencies(clist):
      varassign = tf.assign(var, var_out)
      #varassign = tf.Print(varassign,[],message="varassign")
      
      alist.append(varassign)
      return [isconverged,tf.group(alist)]




class LSR1TrustOBS:
    
  def __init__(self, loss, var,grad, k=100, initialtrustradius = 1.):
    self.k = k
    
    self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    #self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    self.atboundary_old = tf.Variable(False, trainable=False)
    self.doiter_old = tf.Variable(False, trainable = False)
    self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    self.isfirstiter = tf.Variable(True, trainable=False)
    #self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    #self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    self.ST = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False, validate_shape=False)
    self.YT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False, validate_shape=False)
    self.psi = tf.Variable(tf.zeros([var.shape[0],0],dtype=var.dtype),trainable = False, validate_shape=False)
    self.M = tf.Variable(tf.zeros([0,0],dtype=var.dtype),trainable = False, validate_shape=False)
    self.MpsiT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype),trainable = False, validate_shape=False)
    #self.gamma = tf.constant(tf.ones([1],dtype=var.dtype), trainable = False)
    #self.gamma = tf.ones([],dtype=var.dtype)
    self.gamma = tf.Variable(tf.ones([],dtype=var.dtype),trainable=False)
    self.doscaling = tf.Variable(False)
    self.updateidx = tf.Variable(tf.zeros([],dtype=tf.int32),trainable = False)
    #self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
  def initialize(self, loss, var, grad):
    alist = []
    alist.append(tf.assign(self.var_old,var))
    alist.append(tf.assign(self.grad_old,grad))
    
    return tf.group(alist)
    
  #def initialize(self, loss, var, k=7, initialtrustradius = 1.):
    #self.k = k
    
    #self.trustradius = tf.Variable(initialtrustradius*tf.ones_like(loss),trainable=False)
    #self.loss_old = tf.Variable(tf.zeros_like(loss), trainable=False)
    #self.predicted_reduction = tf.Variable(tf.zeros_like(loss), trainable = False)
    #self.var_old = tf.Variable(tf.zeros_like(var),trainable=False)
    #self.atboundary_old = tf.Variable(False, trainable=False)
    #self.doiter_old = tf.Variable(True, trainable = False)
    #self.grad_old = tf.Variable(tf.zeros_like(var), trainable=False)
    #self.isfirstiter = tf.Variable(True, trainable=False)
    ##self.B = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    ##self.H = tf.Variable(tf.eye(int(var.shape[0]),dtype=var.dtype),trainable=False)
    #self.ST = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.YT = tf.Variable(tf.zeros([0,var.shape[0]],dtype=var.dtype), trainable = False)
    #self.psi = tf.Variable(tf.zeros([var.shape[0],0],dtype=var.dtype),trainable = False)
    #self.M = tf.Variable(tf.zeros([0,0],dtype=var.dtype),trainable = False)
    ##self.gamma = tf.constant(tf.ones([1],dtype=var.dtype), trainable = False)
    #self.gamma = tf.ones([1],dtype=var.dtype)
    ##self.doscaling = tf.Variable(True)
    #self.updateidx = tf.Variable(tf.zeros([1],dtype=tf.int32),trainable = False)
    #self.grad = tf.gradients(loss,var, gate_gradients=True)[0]
        
    #alist = []
    #alist.append(tf.assign(self.trustradius,initialtrustradius))
    #alist.append(tf.assign(self.loss_old, loss))
    #alist.append(tf.assign(self.predicted_reduction, 0.))
    #alist.append(tf.assign(self.var_old, var))
    #alist.append(tf.assign(self.atboundary_old,False))
    #alist.append(tf.assign(self.doiter_old,False))
    #alist.append(tf.assign(self.isfirstiter,True))
    #alist.append(tf.assign(self.ST,tf.zeros_like(self.ST))
    #alist.append(tf.assign(self.YT,tf.zeros_like(self.YT))
    ##alist.append(tf.assign(self.doscaling,True))
    #alist.append(tf.assign(self.grad_old,self.grad))
    
    ##if doScaling
    

    #return tf.group(alist)

  
  def minimize(self, loss, var, grad = None):
    
    if grad is None:
      grad = tf.gradients(loss,var, gate_gradients=True)[0]
    
    
    xtol = np.finfo(var.dtype.as_numpy_dtype).eps
    #edmtol = math.sqrt(xtol)
    #edmtol = xtol
    #edmtol = 1e-8
    #edmtol = 0.
    eta = 0.
    #eta = 0.15
    #tau1 = 0.1
    #tau2 = 0.3

    #defaults from nocedal and wright
    #eta = 1e-3
    tau1 = 0.1
    tau2 = 0.75
          
    actual_reduction = self.loss_old - loss
    
    #actual_reduction = tf.Print(actual_reduction,[self.loss_old, loss, actual_reduction])
    isnull = tf.logical_not(self.doiter_old)
    rho = actual_reduction/self.predicted_reduction
    rho = tf.where(tf.is_nan(loss), tf.zeros_like(loss), rho)
    rho = tf.where(isnull, tf.ones_like(loss), rho)
  
    dgrad = grad - self.grad_old
    dx = var - self.var_old
    dxmag = tf.sqrt(tf.reduce_sum(tf.square(dx)))
  
    #trustradius_out = tf.where(tf.less(rho,0.25),0.25*self.trustradius,tf.where(tf.logical_and(tf.greater(rho,0.75),self.atboundary_old),2.*self.trustradius, self.trustradius))
    #trustradius_out = tf.minimum(trustradius_out,1e10)
    
    trustradius_out = tf.where(tf.less(rho,tau1),0.5*self.trustradius,
                               tf.where(tf.less(rho,tau2), self.trustradius,
                               tf.where(tf.less_equal(dxmag,0.8*self.trustradius), self.trustradius,
                               2.*self.trustradius)))
                               
    trustradius_out = tf.where(self.doiter_old, trustradius_out, self.trustradius)

    
    trustradius_out = tf.Print(trustradius_out, [actual_reduction,self.predicted_reduction,rho, trustradius_out], message = "actual_reduction, self.predicted_reduction, rho, trustradius_out: ")
    
    #def hesspexact(v):
      #return tf.gradients(self.grad*tf.stop_gradient(v),var, gate_gradients=True)[0]    
    
    #def hesspapprox(B,v):
      #return tf.reshape(tf.matmul(B,tf.reshape(v,[-1,1])),[-1])    
    
    #def Bv(gamma,psi,M,vcol):
      #return gamma*vcol + tf.matmul(psi,tf.matmul(M,tf.matmul(psi,vcol,transpose_a=True)))
    
    #def Bvflat(gamma,psi,M,v):
      #vcol = tf.reshape(v,[-1,1])
      #return tf.reshape(Bv(gamma,psi,M,vcol),[-1])
      
    def Bv(gamma,psi,MpsiT,vcol):
      return gamma*vcol + tf.matmul(psi,tf.matmul(MpsiT,vcol))
    
    def Bvflat(gamma,psi,MpsiT,v):
      vcol = tf.reshape(v,[-1,1])
      return tf.reshape(Bv(gamma,psi,MpsiT,vcol),[-1])
    
    
    
    def doSR1Scaling(yin,dxin):
      s_norm2 = tf.reduce_sum(tf.square(dxin))
      y_norm2 = tf.reduce_sum(tf.square(yin))
      ys = tf.abs(tf.reduce_sum(yin*dxin))
      invalid = tf.equal(ys,0.) | tf.equal(y_norm2, 0.) | tf.equal(s_norm2, 0.)
      scale = tf.where(invalid, tf.ones_like(ys), y_norm2/ys)
      scale = tf.Print(scale,[scale],message = "doing sr1 scaling")
      return (scale,False)
    
    gamma,doscaling = tf.cond(self.doscaling & self.doiter_old, lambda: doSR1Scaling(dgrad,dx), lambda: (self.gamma,self.doscaling))

    
    def doSR1Update(STin,YTin,yin,dxin):
      ycol = tf.reshape(yin,[-1,1])
      dxcol = tf.reshape(dxin,[-1,1])
      
      yrow = tf.reshape(yin,[1,-1])
      dxrow = tf.reshape(dxin,[1,-1])
      
      #dyBx = ycol - Bv(gamma,self.psi,self.M,dxcol)
      dyBx = ycol - Bv(gamma,self.psi,self.MpsiT,dxcol)
      den = tf.matmul(dyBx, dxcol, transpose_a = True)
      #den = tf.reshape(den,[])
      
      dennorm = tf.sqrt(tf.reduce_sum(tf.square(dx)))*tf.sqrt(tf.reduce_sum(tf.square(dyBx)))
      dentest = tf.greater(tf.abs(den),1e-8*dennorm)
      dentest = tf.reshape(dentest,[])
      nonzero = dentest
      #nonzero = tf.logical_and(dentest,tf.not_equal(actual_reduction,0.))
      
      #nonzero = tf.Print(nonzero, [den,dennorm, dentest, nonzero], message = "den, dennorm, dentest, nonzero")
      
      #nonzero = tf.abs(den) > 1e-8
      
      #doappend = tf.logical_and(nonzero, tf.shape(STin)[0] < self.k)
      #doreplace = tf.logical_and(nonzero, tf.shape(STin)[0] >= self.k)
      
      sliceidx = tf.where(tf.shape(STin)[0] < self.k, 0, 1)
      
      #print(den.shape)
      
      def update():
        ST = tf.concat([STin[sliceidx:],dxrow],axis=0)
        YT = tf.concat([YTin[sliceidx:],yrow],axis=0)
        return (ST,YT)
      
      ST,YT = tf.cond(nonzero, update, lambda: (STin, YTin))

      return (ST,YT)
    
    ST = self.ST
    YT = self.YT
        

    #doscaling = tf.constant(False)
    ST,YT = tf.cond(self.doiter_old, lambda: doSR1Update(ST,YT,dgrad,dx), lambda: (ST,YT))    
    
    #compute compact representation
    S = tf.transpose(ST)
    Y = tf.transpose(YT)
    psi = Y - gamma*S
    psiT = tf.transpose(psi)
    STY = tf.matmul(ST,YT,transpose_b=True)
    D = tf.matrix_band_part(STY,0,0)
    L = tf.matrix_band_part(STY,-1,0) - D
    LT = tf.transpose(L)
    STB0S = gamma*tf.matmul(ST,S)
    Minverse = D + L + LT - STB0S
    MpsiT = tf.matrix_solve(Minverse,psiT)
    #M = tf.matrix_inverse(Minverse)
    
    #psi = tf.Print(psi,[psi],message="psi: ")
    #M = tf.Print(M,[M],message="M: ")
    
    isconvergedxtol = trustradius_out < xtol
    #isconvergededmtol = tf.logical_not(self.isfirstiter) & (self.predicted_reduction <= 0.)
    isconvergededmtol = self.predicted_reduction <= 0.
    
    isconverged = self.doiter_old & (isconvergedxtol | isconvergededmtol)
    
    doiter = tf.logical_and(tf.greater(rho,eta),tf.logical_not(isconverged))
    
    #doiter = tf.Print(doiter, [doiter, isconvergedxtol, isconvergededmtol,isconverged,trustradius_out])
    
    def build_sol():
      #grad = self.grad
      
      #compute eigen decomposition
      #psiTpsi = tf.matmul(psiT,psi)
      #epsiTpsi = tf.self_adjoint_eigvals(psiTpsi)
      #e0psiTpsi = tf.reduce_min(epsiTpsi)
      #psiTpsi = tf.Print(psiTpsi,[e0psiTpsi], message = "e0psiTpsi")
      ##psiTpsi = psiTpsi + 4.*tf.maximum(-e0psiTpsi,tf.zeros_like(e0psiTpsi))*tf.eye(tf.shape(psiTpsi)[0],dtype=psiTpsi.dtype)
      
      #RT = tf.cholesky(psiTpsi)
      #R = tf.transpose(RT)
      
      #def chol():
        #RT = tf.cholesky(psiTpsi)
        #R = tf.transpose(RT)
        #return (R,RT)
      
      #def qr():
        #Q,R = tf.qr(psi)
        #RT = tf.transpose(R)
        #return (R,RT)
      
      ##R,RT = tf.cond(e0psiTpsi > 0., chol, qr)
      #R,RT = chol()
      
      #RT = tf.cholesky(psiTpsi)
      #R = tf.transpose(RT)
      
      
      Q,R = tf.qr(psi)
      detR = tf.matrix_determinant(R)
      #assertR = tf.Assert(tf.not_equal(detR,0.),[detR])
      #with tf.control_dependencies([assertR]):
        #R = tf.Print(R,[detR],message="detR")
      
      RT = tf.transpose(R)
      MRT = tf.matrix_solve(Minverse,RT)
      RMRT = tf.matmul(R,MRT)
      #RMRT = tf.matmul(R,tf.matmul(M,R,transpose_b=True))

      e,U = tf.self_adjoint_eig(RMRT) #TODO: check if what is returned here should actually be UT in the paper
      

      
      #R = tf.Print(R,[detR],message = "detR")

      
      #Rinverse = tf.matrix_inverse(R)
      
      gradcol = tf.reshape(grad,[-1,1])
      
      #gpll = tf.matmul(tf.matmul(psi,tf.matmul(Rinverse,U)), gradcol,transpose_a=True)
      #gpll = tf.matmul(tf.matmul(psi,tf.matrix_solve(R,U)), gradcol,transpose_a=True)
      gpll = tf.matmul(tf.matmul(Q,U), gradcol,transpose_a=True)
      gpll = tf.reshape(gpll,[-1])
      gpllsq = tf.square(gpll)
      gmagsq = tf.reduce_sum(tf.square(grad))
      gpllmagsq = tf.reduce_sum(gpllsq)
      gperpmagsq = gmagsq - gpllmagsq
      gperpmagsq = tf.maximum(gperpmagsq,tf.zeros_like(gperpmagsq))
      gperpmagsq = tf.reshape(gperpmagsq,[1])
      
      #gpll = tf.Print(gpll,[tf.shape(gpll)], message = "gpll shape:")
      a = gpll
      a = tf.concat([a,tf.sqrt(gperpmagsq)],axis=0)
      #a = a[:var.shape[0]]
      asq = tf.square(a)
      
      #lam = e + gamma
      #lam = tf.concat([lam,tf.reshape(gamma,[1])],axis=0)
      lam = tf.pad(e,[[0,1]]) + gamma
      #lam = lam[:var.shape[0]]

      abarindices = tf.where(asq)
      abarsq = tf.gather(asq,abarindices)
      lambar = tf.gather(lam,abarindices)

      abarsq = tf.reshape(abarsq,[-1])
      lambar = tf.reshape(lambar, [-1])
      
      lambar, abarindicesu = tf.unique(lambar)
      abarsq = tf.unsorted_segment_sum(abarsq,abarindicesu,tf.shape(lambar)[0])
      
      abar = tf.sqrt(abarsq)
      
      #abarsq = tf.square(abar)


      #nv = tf.shape(ST)[0]
      #I = tf.eye(int(var.shape[0]),dtype=var.dtype)
      #B = gamma*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #B = tf.Print(B, [B],message="B: ", summarize=1000)
      #efull = tf.self_adjoint_eigvals(B)
      #lam = efull[:1+nv]
      
      e0 = tf.minimum(lam[0],gamma)
      sigma0 = tf.maximum(-e0,tf.zeros([],dtype=var.dtype))
      
      
      #lambar, lamidxs = tf.unique(lam)
      #abarsq = tf.segment_sum(asq,lamidxs)
      
      abarsq = tf.Print(abarsq, [a, abar, lam, lambar,gperpmagsq,gmagsq,gpllmagsq], message = "a,abar,lam,lambar,gperpmagsq,gmagsq, gpllmagsq")
      
      
      def phif(s):        
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        pmag = tf.sqrt(pmagsq)
        phipartial = tf.reciprocal(pmag)
        singular = tf.reduce_any(tf.equal(-s,lambar))
        #singular = tf.logical_or(singular, tf.is_nan(phipartial))
        #singular = tf.logical_or(singular, tf.is_inf(phipartial))
        phipartial = tf.where(singular, tf.zeros_like(phipartial), phipartial)
        phi = phipartial - tf.reciprocal(trustradius_out)
        return phi
      
      def phiphiprime(s):
        phi = phif(s)
        pmagsq = tf.reduce_sum(abarsq/tf.square(lambar+s))
        phiprime = tf.pow(pmagsq,-1.5)*tf.reduce_sum(abarsq/tf.pow(lambar+s,3))
        return (phi, phiprime)
        
      
      phisigma0 = phif(sigma0)
      #usesolu = e0>0. & phisigma0 >= 0.
      usesolu = tf.logical_and(e0>0. , phisigma0 >= 0.)
      usesolu = tf.Print(usesolu,[sigma0,phisigma0,usesolu], message = "sigma0, phisigma0,usesolu: ")

      
      def sigma_sol():
        tol = 1e-8
        maxiter = 50

        sigmainit = tf.reduce_max(tf.abs(a)/trustradius_out - lam)
        sigmainit = tf.maximum(sigmainit,tf.zeros_like(sigmainit))
        phiinit = phif(sigmainit)
        
        sigmainit = tf.Print(sigmainit,[sigmainit,phiinit],message = "sigmainit, phinit: ")

        
        loop_vars = [sigmainit, phiinit, tf.zeros([],dtype=tf.int32)]
        
        def cond(sigma,phi,j):
          return tf.logical_and(phi < -tol, j<maxiter)
        
        def body(sigma,phi,j):   
          #sigma = tf.Print(sigma, [sigma, phi], message = "sigmain, phiin: ")
          phiout, phiprimeout = phiphiprime(sigma)
          sigmaout = sigma - phiout/phiprimeout
          sigmaout = tf.Print(sigmaout, [sigmaout,phiout, phiprimeout], message = "sigmaout, phiout, phiprimeout: ")
          return (sigmaout,phiout,j+1)
          
        sigmaiter, phiiter, jiter = tf.while_loop(cond, body, loop_vars, parallel_iterations=1, back_prop=False)
        
        return sigmaiter
      
      sigma = tf.cond(usesolu, lambda: sigma0, lambda: sigma_sol())
      tau = sigma + gamma
      
      #print(var.shape[0])
      I = tf.eye(int(var.shape[0]),dtype=var.dtype)
      innerinverse = tau*Minverse + tf.matmul(psi,psi,transpose_a=True)
      innerpsiT = tf.matrix_solve(innerinverse,psiT)
      #inner = tf.matrix_inverse(innerinverse)
      #inner2 = tf.matmul(tf.matmul(psi,inner),psi, transpose_b=True)
      inner2 = tf.matmul(psi,innerpsiT)
      p = -tf.matmul(I-inner2, gradcol)/tau
      p = tf.reshape(p,[-1])
      
      magp = tf.sqrt(tf.reduce_sum(tf.square(p)))
      detMinverse = tf.matrix_determinant(Minverse)
      detinnerinverse = tf.matrix_determinant(innerinverse)
      p = tf.Print(p,[magp,detR,detMinverse,detinnerinverse],message = "magp, detR, detMinverse, detinnerinverse")

      #e0val = efull[0]
      #e0val = e0
      
      #Bfull = tau*I + tf.matmul(psi,tf.matmul(M,psi,transpose_b=True))
      #pfull = -tf.matrix_solve(Bfull,tf.reshape(grad,[-1,1]))
      #pfull = tf.reshape(p,[-1])
      #p = pfull

      #p  = tf.Print(p,[e0,e0val,sigma0,sigma,tau], message = "e0, e0val, sigma0, sigma, tau")
      #p  = tf.Print(p,[lam,efull], message = "lam, efull")

      predicted_reduction_out = -(tf.reduce_sum(grad*p) + 0.5*tf.reduce_sum(Bvflat(gamma, psi, MpsiT, p)*p))
      
      return [var+p, predicted_reduction_out, tf.logical_not(usesolu), grad]

    loopout = tf.cond(doiter, lambda: build_sol(), lambda: [self.var_old+0., tf.zeros_like(loss),tf.constant(False),self.grad_old])
    var_out, predicted_reduction_out, atboundary_out, grad_out = loopout
        
    #var_out = tf.Print(var_out,[],message="var_out")
    #loopout[0] = var_out
    
    alist = []
    
    with tf.control_dependencies(loopout):
      oldvarassign = tf.assign(self.var_old,var)
      alist.append(oldvarassign)
      alist.append(tf.assign(self.loss_old,loss))
      alist.append(tf.assign(self.doiter_old, doiter))
      alist.append(tf.assign(self.ST,ST,validate_shape=False))
      alist.append(tf.assign(self.YT,YT,validate_shape=False))
      alist.append(tf.assign(self.psi,psi,validate_shape=False))
      #alist.append(tf.assign(self.M,M,validate_shape=False))
      alist.append(tf.assign(self.MpsiT,MpsiT,validate_shape=False))
      alist.append(tf.assign(self.doscaling,doscaling))
      alist.append(tf.assign(self.grad_old,grad_out))
      alist.append(tf.assign(self.predicted_reduction,predicted_reduction_out))
      alist.append(tf.assign(self.atboundary_old, atboundary_out))
      alist.append(tf.assign(self.trustradius, trustradius_out))
      alist.append(tf.assign(self.isfirstiter,False)) 
      alist.append(tf.assign(self.gamma,gamma)) 
       
    clist = []
    clist.extend(loopout)
    clist.append(oldvarassign)
    with tf.control_dependencies(clist):
      varassign = tf.assign(var, var_out)
      #varassign = tf.Print(varassign,[],message="varassign")
      
      alist.append(varassign)
      return [isconverged,tf.group(alist)]

