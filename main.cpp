////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
// 
////////////////////////////////////////////////////////////////////////

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <list>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "unistd.h"
#include "sgutils.h"

#define GLEW_STATIC


#include "GL/glew.h"
#include "GL/glfw3.h"

#include "cvec.h"
#include "geometrymaker.h"
#include "glsupport.h"
#include "matrix4.h"
#include "ppm.h"
#include "rigtform.h"
#include "arcball.h"
#include "asstcommon.h"
#include "scenegraph.h"
#include "drawer.h"
#include "picker.h"
#include "geometry.h"
#include "mesh.h"

using namespace std; // for string, vector, iostream, and other standard C++ stuff

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.5.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.5. Use GLSL 1.5 unless your system does not support it.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
const bool g_Gl2Compatible = false;

static const float g_frustMinFov = 60.0; // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;  // near plane
static const float g_frustFar = -100.0;  // far plane
static const float g_groundY = -2.0;    // y coordinate of the ground
static const float g_ceilY = 20.0;      // y coordinate of the ceiling
static const float g_roomWidth = 20.0;
static const float g_roomDepth = 1.5 * g_roomWidth;
static const float g_movementBuffer = 1.0; // skyNode always maintains this distance to the walls of the room
static const float g_lineZ = g_roomDepth/4; // you must start before (z>g_lineZ) this line
static const float g_roomMinDim = min(2*g_roomWidth, min(2*g_roomDepth, g_ceilY-g_groundY)),
    g_roomMaxDim = max(2*g_roomWidth, max(2*g_roomDepth, g_ceilY-g_groundY));

static GLFWwindow* g_window;

static int g_windowWidth = 512;
static int g_windowHeight = 512;
static double g_wScale = 1;
static double g_hScale = 1;
static double g_arcballScreenRadius = .25 * min(g_windowWidth, g_windowHeight);
static double g_arcballScale = 1.;

static bool g_mouseClickDown = false; // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static bool g_spaceDown = false; // space state, for middle mouse emulation
static bool g_pickingMode = false;
static double g_mouseClickX, g_mouseClickY; // coordinates for mouse click event

// --------- Materials
static shared_ptr<Material> g_goldDiffuseMat, g_blueDiffuseMat, g_arrowDiffuseMat,
    g_bumpFloorMat, g_arcballMat, g_pickingMat, g_lightMat, g_transMat;
static const Cvec3f gold = Cvec3f(165./255, 124./255, 0./255), blue = Cvec3f(1./255, 1./255, 136./255),
    red = Cvec3f(179./255, 33./255, 19./255);

shared_ptr<Material> g_overridingMaterial;
static vector<shared_ptr<Material>> g_bunnyShellMats; // for bunny shells

// --------- Scene nodes
static shared_ptr<SgTransformNode> g_world, g_groundNode, g_activeEyeNode, g_screenNode, g_arrowEyeNode; 
static shared_ptr<SgRbtNode> g_skyNode, g_arrowNode, g_light1Node, g_light2Node, g_currentPickedRbtNode;
static list<shared_ptr<SgRbtNode>> g_balloonNodes;
static vector<Cvec3> g_balloonVelocity; // in world coordinates

static bool g_skyCameraOrbit = true;

// Is the animation playing?
static bool g_playingAnimation = false;
// Time since last key frame
static int g_animateTime = 0;
static double g_lastFrameClock;
static const double g_framesPerSecond = 60;

// --------- Geometry
typedef SgGeometryShapeNode MyShapeNode;

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;

static bool g_print = true; // for testing only

// -------- For the game

// Global variables for used physical simulation
static const Cvec3 g_gravity(0, -0.5, 0); // gavity vector
static double g_timeStep = 0.02;
static double g_numStepsPerFrame = 10;

// Game status and statistics
static int g_scorePerPop = 10;
static int g_currScore = 0;

static float g_arrowWidth = .3;
static float g_arrowWidthMax = g_roomDepth*.03, g_arrowWidthMin = 0.1;
static float g_arrowLength = g_arrowWidth*10;
static Cvec3 g_arrowVelocity(0,0,0);
static bool g_arrowLock = false;
static bool g_arrowSetPower = false;
static float g_arrowPower = 0.0;

static float g_balloonRadius = 2.0;
static bool g_balloonMoving = false;
static float g_balloonVelocityScale = 5.0;
static const float g_balloonVelocityMin = 0.1, g_balloonVelocityMax = min(60.0, g_roomMinDim/g_timeStep);
static const float g_incrementRatio = 1.2;



///////////////// END OF G L O B A L S
/////////////////////////////////////////////////////

static void initGround() {
    int ibLen, vbLen;
    getPlaneVbIbLen(vbLen, ibLen);

    // Temporary storage for cube Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);

    makePlane(g_roomMaxDim, vtx.begin(), idx.begin());
    g_ground.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initCubes() {
    int ibLen, vbLen;
    getCubeVbIbLen(vbLen, ibLen);

    // Temporary storage for cube Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);

    makeCube(1, vtx.begin(), idx.begin());
    g_cube.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initSphere() {
    int ibLen, vbLen;
    getSphereVbIbLen(20, 10, vbLen, ibLen);

    // Temporary storage for sphere Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);
    makeSphere(1, 20, 10, vtx.begin(), idx.begin());
    g_sphere.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vtx.size(), idx.size()));
}

static Cvec3 coordConvert(Cvec3 p, RigTForm& worldToObject, bool toWorldCoord = true, float translate = 1.) {
    return Cvec3((toWorldCoord ? worldToObject : inv(worldToObject))*Cvec4(p,translate));
}

static Cvec3 limitMotion(RigTForm& L, RigTForm worldToObject = RigTForm(), 
    float buffer = g_movementBuffer,
    float zmin = -g_roomDepth, float zmax = g_roomDepth,
    float xmin = -g_roomWidth, float xmax = g_roomWidth, 
    float ymin = g_groundY, float ymax = g_ceilY) {
    Cvec3 t = coordConvert(L.getTranslation(), worldToObject);
    t[0] = clamp((float)t[0], xmin+buffer, xmax-buffer);
    t[1] = clamp((float)t[1], ymin+buffer, ymax-buffer);
    t[2] = clamp((float)t[2], zmin+buffer, zmax-buffer);
    L.setTranslation(coordConvert(t, worldToObject, false));
    return L.getTranslation();
}

static float angleBetween(Cvec3 v1, Cvec3 v2) {
    if ((norm(v1) > CS175_EPS) && (norm(v2) > CS175_EPS))
        return acos(dot(v1,v2)/norm(v1)/norm(v2));
    else
        return 0;
}

static void printCvec3(Cvec3 p) {
    cout << " " << p[0] << " " << p[1] << " " << p[2] << endl;
}

static Cvec3 getBasis(Quat q, int i) {
  return quatToMatrix(q).getBasis(i);
}

static list<shared_ptr<SgRbtNode>>::iterator popBalloon(
    list<shared_ptr<SgRbtNode>>::iterator balloon,
    vector<Cvec3>::iterator balloonVelocity) {
    g_world->removeChild(*balloon);
    g_currScore += g_scorePerPop;
    cout << "Pop! Current score " << g_currScore << endl;
    g_balloonVelocity.erase(balloonVelocity);
    return g_balloonNodes.erase(balloon);
}

static bool hitWall(Cvec3 p, double tol = g_arrowWidth/2) {
    return ((p[0] < -g_roomWidth + tol) || (p[0] > g_roomWidth - tol) 
        || (p[1] < g_groundY + tol) || (p[1] > g_ceilY - tol) 
        || (p[2] < -g_roomDepth + tol) || (p[2] > g_roomDepth - tol) );
}

static void rescaleArrow() {
    g_arrowNode->clearChild();
    g_arrowLength = g_arrowWidth * 10;
    g_arrowNode->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_cube, g_blueDiffuseMat, Cvec3(0, 0, 0), Cvec3(0,0,0), 
            Cvec3(g_arrowWidth/2, g_arrowWidth/2, g_arrowLength))));
    g_arrowNode->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_sphere, g_blueDiffuseMat, Cvec3(0, 0, -g_arrowLength/2), Cvec3(0,0,0), 
            Cvec3(g_arrowWidth/2, g_arrowWidth/2, g_arrowWidth/2))));
    g_arrowEyeNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0,g_arrowWidth/2,-g_arrowLength/2))));
    g_arrowNode->addChild(g_arrowEyeNode);
}

static void resetScene() {
    if (g_arrowEyeNode) {
        g_arrowNode->removeChild(g_arrowEyeNode);
        g_world->removeChild(g_skyNode);
        g_world->removeChild(g_arrowNode);
    }
    g_skyNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, (g_ceilY+g_groundY)/2, g_roomDepth-g_movementBuffer))));
    g_activeEyeNode = g_skyNode;
    g_arrowNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, (g_ceilY+g_groundY)/2, (g_lineZ + g_roomDepth)/2),
        Quat(cos(0/8),Cvec3(0,1,0)*sin(0/8)))));
    rescaleArrow();

    g_world->addChild(g_arrowNode);
    g_world->addChild(g_skyNode);
    g_arrowVelocity = Cvec3(0,0,0);
    g_arrowLock = false;
    g_arrowPower = 0.0;
}

static void updateArrow() {
    if (norm(g_arrowVelocity) < CS175_EPS)
        return;
    RigTForm worldToArrow = getPathAccumRbt(g_world, g_arrowNode, 1);
    RigTForm L = getPathAccumRbt(g_world, g_arrowNode);
    Cvec3 p = coordConvert(L.getTranslation(), worldToArrow); // arrow COM location in world coordinates
    p = p + g_arrowVelocity * g_timeStep;
    float theta = angleBetween(g_arrowVelocity, g_arrowVelocity + g_gravity * g_timeStep);
    Cvec3 k = coordConvert(Cvec3(-g_arrowVelocity[2],0,g_arrowVelocity[0]), L, false, 0);
    if (theta > CS175_EPS) {
        k.normalize();
    } else {
        theta = 0;
        k = Cvec3(1,0,0);
    }
    g_arrowVelocity = g_arrowVelocity + g_gravity * g_timeStep;
    g_arrowNode->setRbt(RigTForm(coordConvert(p, worldToArrow, false), 
        inv(worldToArrow.getRotation())*L.getRotation()*Quat(cos(theta/2),k*sin(-theta/2))));
    g_world->removeChild(g_groundNode);
    g_world->addChild(g_groundNode);
    L = getPathAccumRbt(g_world, g_arrowNode);
    Cvec3 s = coordConvert(Cvec3(0,0,-g_arrowLength/2), L); // arrow tip location in world coordinates
    // Verify that veloc vector = arrow vector in direction
    // cout << "veloc vector "; printCvec3(g_arrowVelocity/norm(g_arrowVelocity));
    // cout << "arrow vector "; printCvec3(s-p);
    auto it2 = g_balloonVelocity.begin();
    for(auto it = g_balloonNodes.begin(), end = g_balloonNodes.end(); it != end; it++) {
        if (norm2(s - ((*it)->getRbt()).getTranslation()) < g_balloonRadius*g_balloonRadius) {
            it = popBalloon(it, it2);
        }
        it2++;
    }
    s += g_arrowVelocity * g_timeStep;
    if (hitWall(s)) {
        cout << "Arrow hits wall. Your total is "<< g_currScore << endl;
        cout << "Resetting arrow in 3s..." << endl;
        sleep(3);
        resetScene();
    }
    
}

// takes a projection matrix and send to the the shaders
inline void sendProjectionMatrix(Uniforms& uniforms, const Matrix4& projMatrix) {
    uniforms.put("uProjMatrix", projMatrix);
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
    if (g_windowWidth >= g_windowHeight)
        g_frustFovY = g_frustMinFov;
    else {
        const double RAD_PER_DEG = 0.5 * CS175_PI / 180;
        g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight /
                                g_windowWidth,
                            cos(g_frustMinFov * RAD_PER_DEG)) /
                      RAD_PER_DEG;
    }
}

static Matrix4 makeProjectionMatrix() {
    return Matrix4::makeProjection(
        g_frustFovY, g_windowWidth / static_cast<double>(g_windowHeight),
        g_frustNear, g_frustFar);
}

static void updateActiveEye() {
    cout << "Active eye is ";
    if (g_activeEyeNode == g_skyNode) {
        g_activeEyeNode = g_arrowEyeNode;
        cout << "Arrow" << endl;
    } else if (g_activeEyeNode == g_arrowEyeNode) {
        g_activeEyeNode = g_skyNode;
        cout << "Sky" << endl;
    }
}

static RigTForm getEyeRbt() {
    return (getPathAccumRbt(g_world, g_activeEyeNode));
}

// -------- Game

static double randomRange(float lo = -1.0, float hi = 1.0) {
    return(static_cast <double> (lo + (hi-lo) * rand()/RAND_MAX));
}

static Cvec3 randomCvec3(float scale=1.0) {
    float theta = randomRange(0,CS175_PI);
    float phi = randomRange(0,CS175_PI*2);
    return Cvec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))*scale;
}

static void generateBalloon(int n = 1, bool randomLoc = true) {
    shared_ptr<SgRbtNode> balloon;
    Cvec3 loc;
    for (int i = 0; i < n; i++) {
        loc = randomLoc ? Cvec3(
            randomRange(-g_roomWidth+g_balloonRadius,  g_roomWidth-g_balloonRadius), 
            randomRange(g_groundY+g_balloonRadius,    g_ceilY-g_balloonRadius), 
            randomRange(-g_roomDepth+g_balloonRadius,  g_lineZ-g_balloonRadius)) : Cvec3(0.0, g_groundY+g_balloonRadius, g_lineZ-g_balloonRadius);
        balloon.reset(new SgRbtNode(RigTForm(loc)));
        balloon->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_sphere, g_goldDiffuseMat, 
            Cvec3(0, 0, 0), Cvec3(0, 0, 0), Cvec3(g_balloonRadius, g_balloonRadius, g_balloonRadius))));
        g_world->removeChild(g_groundNode);
        g_world->addChild(balloon);
        g_world->addChild(g_groundNode);
        g_balloonNodes.push_back(balloon);
        g_balloonVelocity.push_back(randomCvec3());
    }
}

static void resizeBalloon() {
    int i = g_balloonNodes.size();
    RigTForm L;
    for(auto it = g_balloonNodes.begin(); i > 0; i--, it++) {
        RigTForm L = (*it)->getRbt();
        (*it)->clearChild();
        (*it)->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_sphere, g_goldDiffuseMat, 
            Cvec3(0, 0, 0), Cvec3(0,0,0), Cvec3(g_balloonRadius, g_balloonRadius, g_balloonRadius))));
        limitMotion(L, getPathAccumRbt(g_world,*it,1), g_balloonRadius, -g_roomDepth, g_lineZ);
        (*it)->setRbt(L);
    }
    cout << "Balloon resized. New radius: " << g_balloonRadius << endl;
}

static RigTForm updateBalloon(RigTForm L, Cvec3& vel) {
    Cvec3 pos = L.getTranslation();
    pos += vel * g_timeStep * g_balloonVelocityScale;
    if (pos[0]-g_balloonRadius < -g_roomWidth) {
        pos[0] = -2*g_roomWidth-(pos[0]-g_balloonRadius)+g_balloonRadius;
        vel[0] = -vel[0];
    } else if (pos[0]+g_balloonRadius > g_roomWidth) {
        pos[0] = 2*g_roomWidth-(pos[0]+g_balloonRadius)-g_balloonRadius;
        vel[0] = -vel[0];
    }

    if (pos[1]-g_balloonRadius < g_groundY) {
        pos[1] = 2*g_groundY-(pos[1]-g_balloonRadius)+g_balloonRadius;
        vel[1] = -vel[1];
    } else if (pos[1]+g_balloonRadius > g_ceilY) {
        pos[1] = 2*g_ceilY-(pos[1]+g_balloonRadius)-g_balloonRadius;
        vel[1] = -vel[1];
    }

    if (pos[2]-g_balloonRadius < -g_roomDepth) {
        pos[2] = -2*g_roomDepth-(pos[2]-g_balloonRadius)+g_balloonRadius;
        vel[2] = -vel[2];
    } else if (pos[2]+g_balloonRadius > g_lineZ) {
        pos[2] = 2*g_lineZ-(pos[2]+g_balloonRadius)-g_balloonRadius;
        vel[2] = -vel[2];
    }
    L.setTranslation(pos);
    return L;
} 

static void updateBalloons() {
    if (g_balloonMoving) {
        vector<Cvec3>::iterator it2 = g_balloonVelocity.begin();
        for(auto it = g_balloonNodes.begin(), end = g_balloonNodes.end(); it != end; it++) {
            (*it)->setRbt(updateBalloon((*it)->getRbt(), *it2));
            it2++;
        }
    }
}

// ------- Arcball
// returns object center in eye frame
static Cvec3 getArcballCenter() {
    RigTForm pickedRbt = (g_currentPickedRbtNode) ? getPathAccumRbt(g_world, g_currentPickedRbtNode) : RigTForm();
    Cvec3 center = (inv(getEyeRbt()) * pickedRbt).getTranslation(); 
    //cout << center[0] << " " << center[1] << " " << center[2] << endl; cout.flush();
    return center;
}

static void updateArcballScale() {
    double z = getArcballCenter()[2];
    g_arcballScale = ((z < -CS175_EPS) ? 
        getScreenToEyeScale(z, g_frustFovY, g_windowHeight) : .01);
}

static void drawStuff(bool picking = false) {
    if (!(g_mouseLClickButton && g_mouseRClickButton) && !g_mouseMClickButton)
        updateArcballScale(); 

    Uniforms uniforms;

    // build & send proj. matrix to vshader
    const Matrix4 projmat = makeProjectionMatrix();
    sendProjectionMatrix(uniforms, projmat);

    // use the correct eyeRbt from g_whichView
    const RigTForm eyeRbt = getEyeRbt();
    const RigTForm invEyeRbt = inv(eyeRbt);

    const Cvec3 eyeLight1 = Cvec3(
        invEyeRbt * Cvec4(getPathAccumRbt(g_world, g_light1Node).getTranslation(), 1));
    const Cvec3 eyeLight2 = Cvec3(
        invEyeRbt * Cvec4(getPathAccumRbt(g_world, g_light2Node).getTranslation(), 1));
    uniforms.put("uLight", eyeLight1);
    uniforms.put("uLight2", eyeLight2);

    if (!picking) {
        Drawer drawer(invEyeRbt, uniforms);
        g_world->accept(drawer);

        // draw spheres
        if (g_currentPickedRbtNode && !g_arrowSetPower)  {
            Matrix4 MVM = rigTFormToMatrix(invEyeRbt * getPathAccumRbt(g_world, g_currentPickedRbtNode));
            MVM = MVM * Matrix4::makeScale(Cvec3(g_arcballScale * g_arcballScreenRadius));
            Matrix4 NMVM = normalMatrix(MVM);
            sendModelViewNormalMatrix(uniforms, MVM, NMVM);
            // safe_glUniform3f(uniforms.h_uColor, 155./255, 28./255, 49./255);
            g_arcballMat->draw(*g_sphere, uniforms);
        }
    } else {
        // intialize the picker with our uniforms, as opposed to curSS
        Picker picker(invEyeRbt, uniforms);

        // set overiding material to our picking material
        g_overridingMaterial = g_pickingMat;

        g_world->accept(picker);

        // unset the overriding material
        g_overridingMaterial.reset();

        glFlush();
        // The OpenGL framebuffer uses pixel units, but it reads mouse coordinates
        // using point units. Most of the time these match, but on some hi-res
        // screens there can be a scaling factor.
        g_currentPickedRbtNode = picker.getRbtNodeAtXY(g_mouseClickX * g_wScale,
                                                       g_mouseClickY * g_hScale);

        if ((g_currentPickedRbtNode != g_light1Node) && (g_currentPickedRbtNode != g_light2Node) && (g_currentPickedRbtNode != g_arrowNode))
            g_currentPickedRbtNode = shared_ptr<SgRbtNode>();   // set to NULL
        if ((g_currentPickedRbtNode == g_arrowNode) && g_arrowLock) {
            g_currentPickedRbtNode = shared_ptr<SgRbtNode>();
            cout << "Cannot manipulate arrow after it is launched" << endl;
        }
        if (g_currentPickedRbtNode) {
            cout << "Part picked" << endl;
        } else {
            cout << "No part picked" << endl;
        }
    }
}

static void pick() {
    // We need to set the clear color to black, for pick rendering.
    // so let's save the clear color
    GLdouble clearColor[4];
    glGetDoublev(GL_COLOR_CLEAR_VALUE, clearColor);

    glClearColor(0, 0, 0, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // No more glUseProgram
    drawStuff(true); // no more curSS

    // Uncomment below to see result of the pick rendering pass
    // glfwSwapBuffers();

    //Now set back the clear color
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);

    checkGlErrors();
}

static Cvec3 findArcballQuat(double x, double y, Cvec3 center) {
    Cvec2 arcballScreenCenter = getScreenSpaceCoord(center, makeProjectionMatrix(), 
                g_frustNear, g_frustFovY, g_windowWidth, g_windowHeight);
    Cvec3 v = Cvec3(x - arcballScreenCenter[0], y - arcballScreenCenter[1], 0);
    double rxy2 = pow(v[0], 2) + pow(v[1], 2);
    if (rxy2 < pow(g_arcballScreenRadius, 2)) {
        v[2] = sqrt(pow(g_arcballScreenRadius, 2) - rxy2);
    }
    return v.normalize();
}

static RigTForm findArcballRbt(double dx, double dy, Cvec3 center) {
    if (center[2] > -CS175_EPS) {
        return RigTForm();
    }
    Cvec3 v1 = findArcballQuat(g_mouseClickX, g_mouseClickY, center);
    Cvec3 v2 = findArcballQuat(g_mouseClickX + dx, g_mouseClickY + dy, center);
    return RigTForm(Quat(0.,v2) * Quat(0.,-v1));
}


// -------- Display 

static void updateWHScale() {
    int screen_width, screen_height, pixel_width, pixel_height;
    glfwGetWindowSize(g_window, &screen_width, &screen_height);
    glfwGetFramebufferSize(g_window, &pixel_width, &pixel_height);
    g_wScale = pixel_width / screen_width;
    g_hScale = pixel_height / screen_height;
    cout << g_wScale << " " << g_hScale << endl;

    // cout << screen_width << " " << screen_height << endl;
    // cout << pixel_width << " " << pixel_width << endl;
}

static void display() {
    // No more glUseProgram
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawStuff(false); // no more curSS

    glfwSwapBuffers(g_window);

    checkGlErrors();
}


static void reshape(GLFWwindow * window, const int w, const int h) {
    int width, height;
    glfwGetFramebufferSize(g_window, &width, &height); 
    glViewport(0, 0, width, height);
    g_arcballScreenRadius = .25 * min(g_windowWidth, g_windowHeight);
    g_windowWidth = w;
    g_windowHeight = h;
    cerr << "Size of window is now " << g_windowWidth << "x" << g_windowHeight << endl;
   //set arcball radius here 
   updateFrustFovY();
}

static RigTForm getM(const double dx, const double dy) {
    RigTForm m;
    Cvec3 center = getArcballCenter();
    if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown) { 
        // left button down? (rotate)
        if (!g_currentPickedRbtNode) {
            return RigTForm(Quat::makeXRotation(-dy) * Quat::makeYRotation(dx));
        } else {
            // Cvec3 v1 = findArcballQuat(g_mouseClickX, g_mouseClickY, c);
            // Cvec3 v2 = findArcballQuat(g_mouseClickX + dx, g_mouseClickY + dy, c);
            return findArcballRbt(dx, dy, center);
        }
    } else {
        double movementScale = (!g_currentPickedRbtNode) ? 0.01 : g_arcballScale;
        if (g_mouseRClickButton && !g_mouseLClickButton) { 
            // right button down? (translate)
            return RigTForm(Cvec3(dx, dy, 0) * movementScale);
        } else if (g_mouseMClickButton ||
                   (g_mouseLClickButton && g_mouseRClickButton) ||
                   (g_mouseLClickButton && !g_mouseRClickButton && g_spaceDown)) { 
            // middle or (left and right, or left + space) button down? (zoom in and out)
            return RigTForm(Cvec3(0, 0, -dy) * movementScale);
        } else {
            return RigTForm();
        }
    }
}


static void motion(GLFWwindow *window, double x, double y) {
    const double dx = x - g_mouseClickX;
    const double dy = g_windowHeight - y - 1 - g_mouseClickY;

    // only manipulate if object is in front of me
    // when object = view = cube, cannot rotate but can translate 
    // in intuitive direction, i.e. reversed
    const RigTForm m = getM(dx, dy);
    if (g_mouseClickDown) {
        if (g_currentPickedRbtNode) { // manipulate object 
            if (g_arrowSetPower) {
                g_arrowPower = max(0.0, g_arrowPower -dy*.01);
            } else {
                RigTForm L = g_currentPickedRbtNode -> getRbt();
                RigTForm CL1 = getPathAccumRbt(g_world,g_currentPickedRbtNode, 1);
                RigTForm CL = CL1 * L;
                const RigTForm A = transFact(CL) * linFact(getEyeRbt());
                const RigTForm S_inv = inv(getPathAccumRbt(g_world,g_currentPickedRbtNode,1));
                const RigTForm As = S_inv * A;
                if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown) { // rotate
                    const RigTForm my = getM(dx, 0);
                    const RigTForm mx = getM(0, dy);
                    const RigTForm B = transFact(CL);
                    const RigTForm Bs = S_inv * B;
                    L = doMtoOwrtA(mx, L, As);
                    L = doMtoOwrtA(my, L, Bs);
                } else { // translate
                    L = doMtoOwrtA(m, L, As);
                }
                Cvec3 t = (CL1 * L).getTranslation();
                limitMotion(L, CL1, (g_currentPickedRbtNode == g_arrowNode) ? 
                    g_arrowLength : g_movementBuffer, g_lineZ);
                g_currentPickedRbtNode -> setRbt(L);
            }
        } else { // manipulate sky
            RigTForm L = g_skyNode -> getRbt();
            if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown) {
                const RigTForm mx = RigTForm(Quat::makeXRotation(dy));
                const RigTForm my = RigTForm(Quat::makeYRotation(-dx));
                if (g_skyCameraOrbit) {
                    // orbit around world's origin
                    L = my * L;
                    L = doMtoOwrtA(mx, L, linFact(L));
                } else {
                    // ego motion
                    L = doMtoOwrtA(my, L, transFact(L)) * mx;
                }
            } else {
                L = L * inv(m);
            }
            limitMotion(L);
            g_skyNode -> setRbt(L);
        }
    }
    

    g_mouseClickX = x;
    g_mouseClickY = g_windowHeight - y - 1;
}

static void mouse(GLFWwindow *window, int button, int state, int mods) {
    double x, y;
    glfwGetCursorPos(window, &x, &y);

    g_mouseClickX = x;
    // conversion from window-coordinate-system to OpenGL window-coordinate-system
    g_mouseClickY = g_windowHeight - y - 1;

    g_mouseLClickButton |= (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS);
    g_mouseRClickButton |= (button == GLFW_MOUSE_BUTTON_RIGHT && state == GLFW_PRESS);
    g_mouseMClickButton |= (button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_PRESS);

    g_mouseLClickButton &= !(button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_RELEASE);
    g_mouseRClickButton &= !(button == GLFW_MOUSE_BUTTON_RIGHT && state == GLFW_RELEASE);
    g_mouseMClickButton &= !(button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_RELEASE);

    g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;

    if (g_pickingMode && g_mouseLClickButton) {
        g_pickingMode = false;
        pick();
        cout << "Picking mode is off" << endl;
    }
    //glutPostRedisplay();
}

static void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            exit(0);
        case GLFW_KEY_H:
            cout << " ============== H E L P ==============\n\n"
                 << "h\t\thelp menu\n"
                 << "s\t\tsave screenshot\n"
                 << "f\t\tToggle flat shading on/off.\n"
                 << "o\t\tCycle object to edit\n"
                 << "v\t\tCycle view\n"
                 << "drag left mouse to rotate\n"
                 << endl;
            break;
        case GLFW_KEY_S: 
            glFlush();
            writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
            break;
        case GLFW_KEY_SPACE:
            g_spaceDown = true;
            break;
        case GLFW_KEY_V:
            updateActiveEye();
            break;
        case GLFW_KEY_M:
            g_skyCameraOrbit = not g_skyCameraOrbit;
            cout << "Editing sky eye w.r.t. ";
            if (g_skyCameraOrbit) {
                cout << "world-sky";
            } else {
                cout << "sky-sky";
            }
            cout << " frame" << endl;
            break;
        case GLFW_KEY_P:
            if (g_arrowSetPower) {
                cout << "Picking mode unavailable when setting arrow power" << endl;
            } else {
                g_pickingMode = true;
                cout << "Picking mode is on" << endl;
            }
            break;
        case GLFW_KEY_G:
            generateBalloon();
            cout << "New balloon generated. # Balloons = " << g_balloonNodes.size() << endl;
            break;
        case GLFW_KEY_F:
            generateBalloon(20);
            cout << "New balloons generated. # Balloons = " << g_balloonNodes.size() << endl;
            break;
        case GLFW_KEY_R:
            resetScene();
            cout << "Arrow and sky node reset." << endl;
            break;
        case GLFW_KEY_L:
            cout << "Prepare for launch! Drag down to boost launch power, or drag up to reduce it." << endl;
            g_arrowPower = 0.0;
            g_arrowLock = true;
            g_arrowSetPower = true;
            g_currentPickedRbtNode = g_arrowNode;
            break;
        case GLFW_KEY_A:
            if (g_arrowPower > CS175_EPS) {
                cout << "... and arrow launched! Let's see how this one goes" << endl;
                g_arrowVelocity = -rigTFormToMatrix(getPathAccumRbt(g_world,g_arrowNode)).getBasis(2).normalize()*g_arrowPower;
                g_arrowSetPower = false;
                g_currentPickedRbtNode = shared_ptr<SgRbtNode>();
            } else {
                cout << "Your arrow has no power. Press [L] and drag to adjust power" << endl;
            }
            break;
        case GLFW_KEY_N:
            cout << "Balloon " << (g_balloonMoving ? "paused" : "moving") << endl;
            g_balloonMoving = not g_balloonMoving;
            break;
        case GLFW_KEY_UP:
            if (g_balloonRadius * g_incrementRatio < g_roomMinDim/2) {
                g_balloonRadius *= g_incrementRatio;
                resizeBalloon();
            } else
                cout << "Max balloon size reached" << endl;
            break;
        case GLFW_KEY_DOWN:
            if (g_balloonRadius > g_incrementRatio * g_arrowWidth) {
                g_balloonRadius /= g_incrementRatio;
                resizeBalloon();
            } else
                cout << "Min balloon size reached" << endl;
            break;
        case GLFW_KEY_RIGHT:
            if (g_balloonVelocityScale*g_incrementRatio < g_balloonVelocityMax) {
                g_balloonVelocityScale *= g_incrementRatio;
                cout << "Balloon velocity: " << g_balloonVelocityScale << endl;
            } else
                cout << "Max balloon velocity reached" << endl;
            break;
        case GLFW_KEY_LEFT:
            if (g_balloonVelocityScale > g_balloonVelocityMin*g_incrementRatio) {
                g_balloonVelocityScale /= g_incrementRatio;
                cout << "Balloon velocity: " << g_balloonVelocityScale << endl;
            } else
                cout << "Min balloon velocity reached" << endl;
            break;
        case GLFW_KEY_EQUAL: // <
            if (g_arrowWidth*g_incrementRatio < g_arrowWidthMax) {
                g_arrowWidth *= g_incrementRatio;
                cout << "Arrow width: " << g_arrowWidth << endl;
                rescaleArrow();
            } else
                cout << "Max arrow width reached" << endl;
            break;
        case GLFW_KEY_MINUS: // <
            if (g_arrowWidth > g_arrowWidthMin*g_incrementRatio) {
                g_arrowWidth /= g_incrementRatio;
                cout << "Arrow width: " << g_arrowWidth << endl;
                rescaleArrow();
            } else
                cout << "Min arrow width reached" << endl;
            break;
        /*case GLFW_KEY_C:
            if (g_playingAnimation)
                cout << "Cannot operate when playing animation" << endl;
            else if (g_keyFrames.empty()) {
                cout << "No key frame defined" << endl;
            } else {
                cout << "Loading current key frame [" << g_currentKeyFrameNumber << "] to scene graph" << endl;
                loadFrame();
            }
            break;
        case GLFW_KEY_N:
            if (g_playingAnimation)
                cout << "Cannot operate when playing animation" << endl;
            else {
                createFrame();
            }
            break;
        case GLFW_KEY_U:
            if (g_playingAnimation) {
                cout << "Cannot operate when playing animation" << endl;
                break;
            }
            if (!g_keyFrames.empty()) {
                loadFrame(false);
            } else {
                createFrame();
            }
            cout << "Copying scene graph to current frame [" << g_currentKeyFrameNumber << "]" << endl;
            break;
        case GLFW_KEY_D:
            if (g_playingAnimation) {
                cout << "Cannot operate when playing animation" << endl;
                break;
            }
            if (g_keyFrames.empty()) {
                cout << "Frame list is now EMPTY" << endl;
                break;
            }
            cout << "Deleting current frame [" << g_currentKeyFrameNumber << "]" << endl;
            g_currentKeyFrame = g_keyFrames.erase(g_currentKeyFrame);
            if (g_currentKeyFrameNumber > 0) {
                g_currentKeyFrame--;
                g_currentKeyFrameNumber--;
            }
            if (g_keyFrames.empty()) {
                cout << "No frames defined" << endl;
                g_currentKeyFrameNumber = -1;
            } else {
                loadFrame();
                cout << "Now at frame [" << g_currentKeyFrameNumber << "]" << endl;
            }
            break;
        case GLFW_KEY_W:
            g_oFrameFile.open(g_frameFileName);
            cout << "Writing animation to " << g_frameFileName << endl;
            g_oFrameFile << g_keyFrames.size() << " " << g_currentKeyFrame->size() << endl;
            for(list<vector<RigTForm>>::iterator it = g_keyFrames.begin(), end = g_keyFrames.end();
                it != end; it++) {
                for(vector<RigTForm>::iterator it1 = it->begin(), end1 = it->end();
                    it1 != end1; it1++) {
                    Cvec3 t = it1->getTranslation();
                    Quat q = it1->getRotation();
                    for(int i = 0; i < 3; i++) {
                        g_oFrameFile << t[i] << " ";
                    }
                    g_oFrameFile << endl;
                    for(int i = 0; i < 4; i++) {
                        g_oFrameFile << q[i] << " ";
                    }
                    g_oFrameFile << endl;
                }
            }
            g_oFrameFile.close();
            break;
        case GLFW_KEY_I:
            if (g_playingAnimation) {
                cout << "Cannot operate when playing animation" << endl;
                break;
            }
            g_iFrameFile.open(g_frameFileName);
            if (g_iFrameFile.fail()) {
                cerr << "Exception caught: Cannot load " << g_frameFileName << endl;
                exit(0);
            }
            cout << "Reading animation from " << g_frameFileName << endl;
            int numFrames, numNodes;
            if (!(g_iFrameFile >> numFrames >> numNodes)) {
                cerr << "Incorrect frame input format! Must start with number of frames." << endl;
                break;
            } else if (numNodes != g_rbtNodes.size()) {
                cerr << "Number of Rbt per frame in " << g_frameFileName << " does not match number of SgRbtNodes in the current scene graph." << endl;
                numNodes = g_rbtNodes.size();
            }
            g_keyFrames.clear();
            cout << numFrames << " frames read." << endl;
            if (numFrames > 0) {
                for(int i = 0; i < numFrames; i++) {
                    vector<RigTForm> tempRbts;
                    for(int j = 0; j < numNodes; j++) {
                        RigTForm temp;
                        Cvec3 t, v;
                        double w;
                        for(int i = 0; i < 3; i++) {
                            g_iFrameFile >> t[i];
                        }
                        g_iFrameFile >> w;
                        for(int i = 0; i < 3; i++) {
                            g_iFrameFile >> v[i];
                        }
                        temp.setTranslation(t);
                        temp.setRotation(Quat(w,v));
                        tempRbts.push_back(temp);
                    }
                    g_keyFrames.push_back(tempRbts);
                }
                g_currentKeyFrameNumber = 0;
                cout << "Now at frame [" << g_currentKeyFrameNumber << "]" << endl;
                g_currentKeyFrame = g_keyFrames.begin();
                loadFrame();
            } else {
                g_currentKeyFrameNumber = -1;
                g_currentKeyFrame = g_keyFrames.begin();
            }
            g_iFrameFile.close();
            break;
        case GLFW_KEY_Y:
            if (g_playingAnimation) {
                endAnimation();
            } else if (g_keyFrames.size() > 3) {
                g_playingAnimation = true;
                g_currentKeyFrame = g_keyFrames.begin();g_currentKeyFrame++; 
                g_currentKeyFrameNumber = 1;
                cout << "Playing animation..." << endl;
            } else {
                cerr << "Cannot play animation with less than 4 keyframes." << endl;
            }
            break;
        case GLFW_KEY_PERIOD: // >
            if (!(mods & GLFW_MOD_SHIFT)) break;
            if (g_playingAnimation) {
                cout << "Cannot operate when playing animation" << endl;
                break;
            }
            g_currentKeyFrame++;
            if (g_currentKeyFrame != g_keyFrames.end()) {
                g_currentKeyFrameNumber += 1;
                cout << "Stepped forward to frame [" << g_currentKeyFrameNumber << "]" << endl;
                loadFrame();
            } else {
                g_currentKeyFrame--;
            }
            break;
        case GLFW_KEY_COMMA: // <
            if (!(mods & GLFW_MOD_SHIFT)) break;
            if (g_playingAnimation) {
                cout << "Cannot operate when playing animation" << endl;
                break;
            }
            if (g_currentKeyFrame != g_keyFrames.begin()) {
                g_currentKeyFrame--;
                g_currentKeyFrameNumber -= 1;
                cout << "Stepped backward to frame [" << g_currentKeyFrameNumber << "]" << endl;
                loadFrame();
            }
            break;
        case GLFW_KEY_EQUAL: // <
            if (!(mods & GLFW_MOD_SHIFT)) break;
            if (g_msBetweenKeyFrames > g_msBetweenKeyFramesMin) {
                g_msBetweenKeyFrames -= g_msBetweenKeyFramesInc;
            }
            cout << g_msBetweenKeyFrames << " ms between keyframs." << endl;
            break;
        case GLFW_KEY_MINUS: // <
            if (g_msBetweenKeyFrames < g_msBetweenKeyFramesMax) {
                g_msBetweenKeyFrames += g_msBetweenKeyFramesInc;
            }
            cout << g_msBetweenKeyFrames << " ms between keyframs." << endl;
            break;
        case GLFW_KEY_RIGHT:
            g_furHeight *= 1.05;
            cerr << "fur height = " << g_furHeight << std::endl;
            updateShellGeometry();
            break;
        case GLFW_KEY_LEFT:
            g_furHeight /= 1.05;
            std::cerr << "fur height = " << g_furHeight << std::endl;
            updateShellGeometry();
            break;
        case GLFW_KEY_UP:
            g_hairyness *= 1.05;
            cerr << "hairyness = " << g_hairyness << std::endl;
            updateShellGeometry();
            break;
        case GLFW_KEY_DOWN:
            g_hairyness /= 1.05;
            cerr << "hairyness = " << g_hairyness << std::endl;
            updateShellGeometry();
            break;*/
        }
    } else {
        switch (key) {
        case GLFW_KEY_SPACE:
            g_spaceDown = false;
            break;
        }
    }
}

void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

static void initGlfwState() {
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

    g_window = glfwCreateWindow(g_windowWidth, g_windowHeight,
                                "Pop the Balloons!", NULL, NULL);
    if (!g_window) {
        fprintf(stderr, "Failed to create GLFW window or OpenGL context\n");
        exit(1);
    }
    glfwMakeContextCurrent(g_window);

    updateWHScale();

    glfwSwapInterval(1);

    glfwSetErrorCallback(error_callback);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetWindowSizeCallback(g_window, reshape);
    glfwSetKeyCallback(g_window, keyboard);

}

static void initGLState() {
    glClearColor(205. / 255., 189. / 255., 162. / 255., 0.);
    glClearDepth(0.);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glReadBuffer(GL_BACK);
    if (!g_Gl2Compatible)
        glEnable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

static void initMaterials() {
    // Create some prototype materials
    Material diffuse("./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader");
    Material solid("./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader");

    // copy diffuse prototype and set red color
    g_goldDiffuseMat.reset(new Material(diffuse));
    g_goldDiffuseMat->getUniforms().put("uColor", Cvec3f(165./255, 124./255, 0./255));

    // copy diffuse prototype and set blue color
    g_blueDiffuseMat.reset(new Material(diffuse));
    g_blueDiffuseMat->getUniforms().put("uColor", Cvec3f(1./255, 1./255, 136./255));

    // normal mapping material
    g_bumpFloorMat.reset(new Material("./shaders/normal-gl3.vshader", "./shaders/normal-gl3.fshader"));
    g_bumpFloorMat->getUniforms().put("uTexColor", shared_ptr<ImageTexture>(new ImageTexture("Fieldstone.ppm", true)));
    g_bumpFloorMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("FieldstoneNormal.ppm", false)));

    // copy solid prototype, and set to wireframed rendering
    g_arcballMat.reset(new Material(solid));
    g_arcballMat->getUniforms().put("uColor", Cvec3f(0.27f, 0.82f, 0.35f));
    g_arcballMat->getRenderStates().polygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // copy solid prototype, and set to color white
    g_lightMat.reset(new Material(solid));
    g_lightMat->getUniforms().put("uColor", Cvec3f(1, 1, 1));

    // pick shader
    g_pickingMat.reset(new Material("./shaders/basic-gl3.vshader", "./shaders/pick-gl3.fshader"));

    // transparent screen shader
    g_transMat.reset(new Material("./shaders/basic-gl3.vshader", "./shaders/transparent-gl3.fshader"));
    g_transMat->getUniforms().put("uColor", Cvec4f(0.50f, 0.f, 0.f, 0.2f));
    // g_transMat->getUniforms().put("uAlpha", 0.0f);
};


static void initGeometry() {
    initGround();
    initCubes();
    initSphere();
}

static void initScene() {
    g_world.reset(new SgRootNode());
    resetScene();
    g_balloonRadius = g_movementBuffer;
    g_currScore = 0;

    g_groundNode.reset(new SgRbtNode());
    g_groundNode->addChild(shared_ptr<MyShapeNode>(
                               new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(0, g_groundY, 0))));
    g_groundNode->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(0, g_ceilY, 0), Cvec3(90, 90, 90))));
    g_groundNode->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(-g_roomWidth, (g_ceilY+g_groundY)/2, 0), Cvec3(0, 0, -90))));
    g_groundNode->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(g_roomWidth, (g_ceilY+g_groundY)/2, 0), Cvec3(0, 0, 90))));
    g_groundNode->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(0, (g_ceilY+g_groundY)/2, -g_roomDepth), Cvec3(90, 0, 0))));
    g_groundNode->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(0, (g_ceilY+g_groundY)/2, g_roomDepth), Cvec3(-90, 0, 0))));
    g_groundNode->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_ground, g_transMat, Cvec3(0, (g_ceilY+g_groundY)/2, g_lineZ), Cvec3(90, 0, 0))));
    g_groundNode->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_ground, g_transMat, Cvec3(0, (g_ceilY+g_groundY)/2, g_lineZ), Cvec3(-90, 0, 0))));
    g_light1Node.reset(new SgRbtNode(RigTForm(Cvec3(-g_roomWidth+g_movementBuffer, g_ceilY-g_movementBuffer, g_lineZ+g_movementBuffer))));
    g_light1Node->addChild(shared_ptr<MyShapeNode>(
                               new MyShapeNode(g_sphere, g_lightMat, Cvec3(0, 0, 0))));
    g_light2Node.reset(new SgRbtNode(RigTForm(Cvec3(g_roomWidth-g_movementBuffer, g_ceilY-g_movementBuffer, g_lineZ+g_movementBuffer))));
    g_light2Node->addChild(shared_ptr<MyShapeNode>(
                               new MyShapeNode(g_sphere, g_lightMat, Cvec3(0, 0, 0))));
    // g_balloonNode.reset(new SgRbtNode(RigTForm(Cvec3(0,0,0))));
    // g_balloonNode->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_sphere, g_goldDiffuseMat, Cvec3(0, 0, 0))));
    
    // g_world->addChild(g_balloonNode);
    g_world->addChild(g_light1Node);
    g_world->addChild(g_light2Node);
    g_world->addChild(g_groundNode);
}

static void glfwLoop() {
    g_lastFrameClock = glfwGetTime();

    while (!glfwWindowShouldClose(g_window)) {
        double thisTime = glfwGetTime();
        if( thisTime - g_lastFrameClock >= 1. / g_framesPerSecond) {

            // animationUpdate();
            // hairsSimulationUpdate();
            updateArrow();
            updateBalloons();
            display();
            g_lastFrameClock = thisTime;
        }
        glfwPollEvents();
    }
}

int main(int argc, char *argv[]) {
    try {
        initGlfwState();

        glewInit(); // load the OpenGL extensions
#ifndef __MAC__

        if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
            throw runtime_error("Error: card/driver does not support OpenGL "
                                "Shading Language v1.3");
        else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
            throw runtime_error("Error: card/driver does not support OpenGL "
                                "Shading Language v1.0");
#endif

        cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0"
                                 : "Will use OpenGL 3.x / GLSL 1.5")
             << endl;

        initGLState();
        initMaterials();
        initGeometry();
        initScene();

        glfwLoop();
        return 0;
    } catch (const runtime_error &e) {
        cout << "Exception caught: " << e.what() << endl;
        return -1;
    }
}
