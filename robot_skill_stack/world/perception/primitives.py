from __future__ import annotations
import numpy as np
from robot_skill_stack.world.model.primitives import PrimitiveGeometry, PrimitiveShape

class PrimitiveEstimator:
    def __init__(self, classification_threshold=.58, ambiguity_margin=.07, top_band=.15):
        self.threshold=float(classification_threshold); self.margin=float(ambiguity_margin); self.top_band=float(top_band)
    @staticmethod
    def _pca(x):
        c=x.mean(0); w,v=np.linalg.eigh(np.cov((x-c).T)); order=np.argsort(w)[::-1]; return c,w[order],v[:,order]
    @staticmethod
    def _yaw(v): return float(np.arctan2(v[1],v[0])%(np.pi/2))
    def estimate(self,candidate):
        p=np.asarray(candidate.metadata.get('points',[]),float)
        if p.ndim!=2 or len(p)<20: return PrimitiveGeometry(PrimitiveShape.UNKNOWN)
        c,e,V=self._pca(p); lo=np.percentile(p,2,axis=0); hi=np.percentile(p,98,axis=0); ext=hi-lo
        zspan=max(ext[2],1e-6); top=p[p[:,2]>=hi[2]-self.top_band*zspan]
        top_flat=np.exp(-np.std(top[:,2])/max(.06*zspan,1e-5)) if len(top)>5 else 0.
        top_mass=float(np.mean(p[:,2]>=hi[2]-.03*zspan))
        planar_top=float(np.clip(top_mass/.12,0,1))
        # Sphere: radial residual is much stronger evidence than equal AABB dimensions.
        r=np.linalg.norm(p-c,axis=1); r0=np.median(r); sphere_res=np.median(np.abs(r-r0))/max(r0,1e-6)
        equal=np.min(ext)/max(np.max(ext),1e-6); sphere=float(np.clip(equal*(1-sphere_res/.18)*(1-.55*top_flat),0,1))
        # Horizontal PCA gives oriented footprint for boxes and lying cylinders.
        xy=p[:,:2]-p[:,:2].mean(0); ew,ev=np.linalg.eigh(np.cov(xy.T)); d=ev[:,np.argmax(ew)]; u=xy@d; q=xy@np.array([-d[1],d[0]])
        rect_fill=(np.ptp(u)*np.ptp(q)); hull_fill=min(1., len(p)/max(len(p),1)) # neutral; depth sees surfaces, not volume
        aspect=max(np.ptp(u),np.ptp(q))/max(min(np.ptp(u),np.ptp(q)),1e-6)
        cube=float(np.clip(planar_top*equal*(1-.70*sphere),0,1))
        vertical=ext[2]>=1.25*max(ext[0],ext[1])
        horizontal=aspect>=1.45 and ext[2]<=.8*max(ext[0],ext[1])
        cyl=0.; axis=None; length=None; radius=None
        if vertical:
            axis=np.array([0.,0.,1.]); length=float(ext[2]); radius=float(np.mean(ext[:2])/2); cyl=float(np.clip(top_flat*.55+(1-abs(ext[0]-ext[1])/max(ext[0],ext[1]))*.45,0,1))
        elif horizontal:
            axis=np.array([d[0],d[1],0.]); length=float(max(np.ptp(u),np.ptp(q))); radius=float(np.median([min(np.ptp(u),np.ptp(q)),ext[2]])/2); cyl=float(np.clip((1-top_flat)*.55+min(1,(aspect-1)/1.5)*.45,0,1))
        scores={'cube':cube,'sphere':sphere,'cylinder':cyl}; ranked=sorted(scores.items(),key=lambda x:x[1],reverse=True); name,score=ranked[0]
        if score<self.threshold or score-ranked[1][1]<self.margin: return PrimitiveGeometry(PrimitiveShape.UNKNOWN,score,scores=scores)
        if name=='sphere': return PrimitiveGeometry(PrimitiveShape.SPHERE,score,radius=float(r0),scores=scores)
        if name=='cylinder': return PrimitiveGeometry(PrimitiveShape.CYLINDER,score,axis=axis,radius=radius,length=length,scores=scores)
        angles=np.linspace(0,np.pi/2,181,endpoint=False); best=(1e9,0.,None)
        for a in angles:
            ca,sa=np.cos(a),np.sin(a); uv=xy@np.array([[ca,-sa],[sa,ca]])
            dims=np.percentile(uv,98,axis=0)-np.percentile(uv,2,axis=0); area=float(dims[0]*dims[1])
            if area<best[0]: best=(area,a,dims)
        yaw=float(best[1]%(np.pi/2)); box=np.array([best[2][0],best[2][1],ext[2]])
        return PrimitiveGeometry(PrimitiveShape.CUBE,score,box_size=box,yaw=yaw,scores=scores)
