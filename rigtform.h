#ifndef RIGTFORM_H
#define RIGTFORM_H

#include <iostream>
#include <cassert>

#include "matrix4.h"
#include "quat.h"

class RigTForm {
  Cvec3 t_; // translation component
  Quat r_;  // rotation component represented as a quaternion

public:
  RigTForm() : t_(0) {
    assert(norm2(Quat(1,0,0,0) - r_) < CS175_EPS2);
  }

  RigTForm(const Cvec3& t, const Quat& r) {
    t_ = t;
    r_ = r;
  }

  explicit RigTForm(const Cvec3& t) {
    t_ = t;
    assert(norm2(Quat(1,0,0,0) - r_) < CS175_EPS2);
  }

  explicit RigTForm(const Quat& r) {
    r_ = r;
  }

  Cvec3 getTranslation() const {
    return t_;
  }

  Quat getRotation() const {
    return r_;
  }

  RigTForm& setTranslation(const Cvec3& t) {
    t_ = t;
    return *this;
  }

  RigTForm& setRotation(const Quat& r) {
    r_ = r;
    return *this;
  }

  Cvec4 operator * (const Cvec4& a) const {
    Cvec3 b(r_*Cvec3(a[0],a[1],a[2]));
    if (abs(a[3] - 1)<CS175_EPS2) {
      b += t_;
    }
    return Cvec4(b, a[3]);
  }

  RigTForm operator * (const RigTForm& a) const {
    return RigTForm(t_ + r_*a.getTranslation(), r_*a.getRotation());
  }

  std::string to_string() {
    std::string s = "";
    for(int i = 0; i < 3; i++) {
      s += std::to_string(t_(i));
    }
    std::cout << s;
    return s;
    //return (to_string(tform.getTranslation()) + to_string(tform.getRotation().q_));
  }
};

inline RigTForm inv(const RigTForm& tform) {
  return RigTForm(-(inv(tform.getRotation())*tform.getTranslation()), 
    inv(tform.getRotation()));
}

inline RigTForm transFact(const RigTForm& tform) {
  return RigTForm(tform.getTranslation());
}

inline RigTForm linFact(const RigTForm& tform) {
  return RigTForm(tform.getRotation());
}

inline RigTForm doMtoOwrtA(const RigTForm& m, const RigTForm& o, 
    const RigTForm& a) {
    return a * m * inv(a) * o;
}

inline Matrix4 rigTFormToMatrix(const RigTForm& tform) {
  return (quatToMatrix(tform.getRotation())
    +Matrix4::makeTranslation(tform.getTranslation())-Matrix4());
}

#endif
