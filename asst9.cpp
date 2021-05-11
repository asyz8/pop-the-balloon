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
static const float g_frustFar = -50.0;  // far plane
static const float g_groundY = -2.0;    // y coordinate of the ground
static const float g_groundSize = 10.0; // half the ground length

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
static bool g_shellNeedsUpdate = true;
// --------- Materials
static shared_ptr<Material> g_redDiffuseMat,
                            g_blueDiffuseMat,
                            g_bumpFloorMat,
                            g_arcballMat,
                            g_pickingMat,
                            g_lightMat,
                            g_bunnyMat;

shared_ptr<Material> g_overridingMaterial;
static vector<shared_ptr<Material>> g_bunnyShellMats; // for bunny shells

// --------- Scene nodes
static shared_ptr<SgTransformNode> g_world, g_groundNode, g_activeEyeNode; 
static shared_ptr<SgRbtNode> g_skyNode, g_robot1Node, g_robot2Node,
    g_light1Node, g_light2Node, g_currentPickedRbtNode, g_bunnyNode;


static list<vector<RigTForm>> g_keyFrames;
static list<vector<RigTForm>>::iterator g_currentKeyFrame = g_keyFrames.begin();
static vector<shared_ptr<SgRbtNode>> g_rbtNodes;
static int g_currentKeyFrameNumber = -1;
static string g_frameFileName = "animation.txt";
static ofstream g_oFrameFile;
static ifstream g_iFrameFile;
static bool g_skyCameraOrbit = true;

// Frames to render per second
static int g_framesPerSecond = 60;
// 2 seconds between keyframes
static int g_msBetweenKeyFrames = 2000;
static int g_msBetweenKeyFramesInc = 100, g_msBetweenKeyFramesMin = 100, g_msBetweenKeyFramesMax = 10000;

// Is the animation playing?
static bool g_playingAnimation = false;
// Time since last key frame
static int g_animateTime = 0;
static double g_lastFrameClock;

// --------- Geometry
typedef SgGeometryShapeNode MyShapeNode;

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;
static const int g_numShells = 24; // constants defining how many layers of shells
static double g_furHeight = 0.21;
static double g_hairyness = 0.7;

static shared_ptr<SimpleGeometryPN> g_bunnyGeometry;
static vector<shared_ptr<SimpleGeometryPNX>> g_bunnyShellGeometries;
static Mesh g_bunnyMesh;

// Global variables for used physical simulation
static const Cvec3 g_gravity(0, -0.5, 0); // gavity vector
static double g_timeStep = 0.02;
static double g_numStepsPerFrame = 10;
static double g_damping = 0.96;
static double g_stiffness = 4;
static int g_simulationsPerSecond = 60;

static std::vector<Cvec3>
    g_tipPos,      // should be hair tip pos in world-space coordinates
    g_tipVelocity; // should be hair tip velocity in world-space coordinates

///////////////// END OF G L O B A L S
/////////////////////////////////////////////////////

static void initGround() {
    int ibLen, vbLen;
    getPlaneVbIbLen(vbLen, ibLen);

    // Temporary storage for cube Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);

    makePlane(g_groundSize*2, vtx.begin(), idx.begin());
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

static Cvec3 coordConvert(Cvec3 p, RigTForm& worldToBunny, bool toWorldCoord = true) {
    return Cvec3((toWorldCoord ? worldToBunny : inv(worldToBunny))*Cvec4(p,1));
}

// finds s in world coordinates
static Cvec3 findMeshRestPos(const Mesh::Vertex& v, RigTForm& worldToBunny) {
    Cvec3 pos = v.getPosition() + v.getNormal() * g_furHeight;
    return coordConvert(pos, worldToBunny);
}

static void initBunnyMeshes() {
    g_bunnyMesh.load("bunny.mesh");

    // Init the per vertex normal of g_bunnyMesh
    int numVert = g_bunnyMesh.getNumVertices(), numFace = g_bunnyMesh.getNumFaces();

    for (int i = 0; i < numVert; ++i) {
        const Mesh::Vertex v = g_bunnyMesh.getVertex(i);
        Mesh::VertexIterator it(v.getIterator()), it0(it);
        Cvec3 norm; 
        do {
            norm += it.getFace().getNormal();
        } while(++it != it0);
        v.setNormal(norm.normalize());
    }

    // cout << "Vertex normal set!" << endl; cout.flush(); 

    // Initialize g_bunnyGeometry from g_bunnyMesh
    // cout << "num vertices should be" << (numFace*3) << endl;
    vector<VertexPN> vertices;
    for (int i = 0; i < numFace; i++) {
        const Mesh::Face f = g_bunnyMesh.getFace(i);
        for (int j = 0; j < f.getNumVertices(); j++) {
            const Mesh::Vertex v = f.getVertex(j);
            vertices.push_back(VertexPN(v.getPosition(), v.getNormal()));
        }
    }
    g_bunnyGeometry.reset(new SimpleGeometryPN(&vertices[0], vertices.size()));

    // Now allocate array of SimpleGeometryPNX to for shells, one per layer
    g_bunnyShellGeometries.resize(g_numShells);
    for (int i = 0; i < g_numShells; ++i) {
        g_bunnyShellGeometries[i].reset(new SimpleGeometryPNX());
    }
}

// Specifying shell geometries based on g_tipPos, g_furHeight, and g_numShells.
// You need to call this function whenver the shell needs to be updated
static void updateShellGeometry() {
    // TASK 1 and 3 TODO: finish this function as part of Task 1 and Task 3
    g_shellNeedsUpdate = false;
    RigTForm worldToBunny = getPathAccumRbt(g_world, g_bunnyNode);
    int numVert = g_bunnyMesh.getNumVertices(), numFace = g_bunnyMesh.getNumFaces();

    if (g_tipPos.empty()) {
        for (int i = 0; i < numVert; i++) {
            const Mesh::Vertex v = g_bunnyMesh.getVertex(i);
            g_tipPos.push_back(findMeshRestPos(v, worldToBunny));
            g_tipVelocity.push_back(Cvec3());
        }
    }

    vector<Cvec3> p, n, d;
    Cvec3 localTip;
    double shellHeight = g_furHeight / g_numShells, mscale = 2.0 / g_numShells / (g_numShells - 1);
    for (int i = 0; i < numVert; i++) {
        const Mesh::Vertex v = g_bunnyMesh.getVertex(i);
        localTip = coordConvert(g_tipPos[i], worldToBunny, false);
        d.push_back((localTip - v.getPosition() - v.getNormal() * g_furHeight) * mscale);
        p.push_back(v.getPosition());
        n.push_back(v.getNormal() * shellHeight);
    }

    vector<Cvec2> t = {Cvec2(0,0), Cvec2(g_hairyness,0), Cvec2(0,g_hairyness)};
    vector<VertexPNX> vertices;
    for (int s = 0; s < g_numShells; ++s) {
        vertices.clear();
        for(int i = 0; i < numVert; i++) {
            p[i] += n[i];
            n[i] += d[i];
        }
        for (int i = 0; i < g_bunnyMesh.getNumFaces(); i++) {
            const Mesh::Face f = g_bunnyMesh.getFace(i);
            for (int j = 0; j < f.getNumVertices(); j++) {
                const int vi = f.getVertex(j).getIndex();
                vertices.push_back(VertexPNX(p[vi], n[vi], t[j]));
            }
        }
        g_bunnyShellGeometries[s]->upload(&vertices[0], vertices.size());
    }
    
}

// New function to update the simulation every frame
static void hairsSimulationUpdate() {
    // TASK 2 TODO: write dynamics simulation code here
    g_shellNeedsUpdate = true;
    int numVert = g_bunnyMesh.getNumVertices(), numFace = g_bunnyMesh.getNumFaces();
    
    RigTForm worldToBunny = getPathAccumRbt(g_world, g_bunnyNode);

    Cvec3 p,s;
    for (int i = 0; i < numVert; i++) {
        const Mesh::Vertex v = g_bunnyMesh.getVertex(i);
        p = coordConvert(v.getPosition(), worldToBunny);
        s = coordConvert(v.getPosition() + 
            v.getNormal() * g_furHeight, worldToBunny);
        for (int t = 0; t < g_numStepsPerFrame; t++) {
            Cvec3 force = g_gravity + (s - g_tipPos[i]) * g_stiffness;
            g_tipPos[i] = g_tipPos[i] + g_tipVelocity[i] * g_timeStep;
            g_tipPos[i] = p + (g_tipPos[i] - p).normalize() * g_furHeight;
            g_tipVelocity[i] = (g_tipVelocity[i] + force * g_timeStep) * g_damping;
        }
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
        g_activeEyeNode = g_robot1Node;
        cout << "Object 1" << endl;
    } else if (g_activeEyeNode == g_robot1Node) {
        g_activeEyeNode = g_robot2Node;
        cout << "Object 2" << endl;
    } else if (g_activeEyeNode == g_robot2Node) {
        g_activeEyeNode = g_skyNode;
        cout << "Sky" << endl;
    }
}

static RigTForm getEyeRbt() {
    return (getPathAccumRbt(g_world, g_activeEyeNode));
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


// ------ Frames
static void initFrame() {
    dumpSgRbtNodes(g_world, g_rbtNodes);
}

// load current key frame to scene graph if such exists
static void loadFrame(bool keyFrameToScene = true) {
    if (g_keyFrames.empty()) {
        return;
    }
    vector<RigTForm>::iterator iter2 = g_currentKeyFrame->begin();
    for (vector<shared_ptr<SgRbtNode>>::iterator iter = g_rbtNodes.begin(), 
        end = g_rbtNodes.end(); iter != end; ++iter) {
        if (keyFrameToScene) 
            (*iter)->setRbt(*iter2);
        else {
            (*iter2) = (*iter) -> getRbt();
        }
        iter2++;
    }
}

// create a vector that stores the current scene graph
static vector<RigTForm> makeFrame() {
    vector<RigTForm> frame;
    for (vector<shared_ptr<SgRbtNode>>::iterator iter = g_rbtNodes.begin(), 
        end = g_rbtNodes.end(); iter != end; ++iter) {
        frame.push_back((*iter)->getRbt());
    }
    return frame;
}

static void createFrame() {
    if (!g_keyFrames.empty()) {
        g_currentKeyFrame++;
    }
    g_currentKeyFrameNumber++;
    g_keyFrames.insert(g_currentKeyFrame, makeFrame());
    g_currentKeyFrame--;
    cout << "Create new frame [" << g_currentKeyFrameNumber << "]" << endl;
}

// For computing d and e
static RigTForm bezierInterpolate3(RigTForm c0, RigTForm c1, RigTForm c2, double alpha) {
    RigTForm rbt;
    rbt.setTranslation((c2.getTranslation() - c0.getTranslation())*alpha 
        + c1.getTranslation());
    rbt.setRotation((((c2.getRotation())
        *inv(c0.getRotation()))^alpha)*(c1.getRotation()));
    return rbt;
}

// For computing other interpolations
static RigTForm bezierInterpolate(RigTForm c0, RigTForm c1, double alpha) {
    return bezierInterpolate3(c0, c0, c1, alpha);
}

static RigTForm catmullRom(vector<RigTForm>::iterator c0, RigTForm d0,
    RigTForm e0, vector<RigTForm>::iterator c1, double alpha) {
    RigTForm f = bezierInterpolate(*c0, d0, alpha);
    RigTForm g = bezierInterpolate(d0, e0, alpha);
    RigTForm h = bezierInterpolate(e0, *c1, alpha);
    RigTForm m = bezierInterpolate(f, g, alpha);
    RigTForm n = bezierInterpolate(g, h, alpha);
    return bezierInterpolate(m, n, alpha);
}

// we use g_currentKeyFrame to keep track of last frame before the current interpolation
bool interpolate(double t) {
    if (t >= g_currentKeyFrameNumber) {
        g_currentKeyFrame++; 
        g_currentKeyFrameNumber++;
    }
    if (g_currentKeyFrameNumber >= (g_keyFrames.size() - 2)) {
        return true;
    }
    static list<vector<RigTForm>>::iterator nextFrame, pprevFrame, nnextFrame;
    nextFrame = g_currentKeyFrame; nextFrame++;
    pprevFrame = g_currentKeyFrame; pprevFrame--;
    nnextFrame = nextFrame; nnextFrame++;
    double alpha = t - floor(t);
    vector<shared_ptr<SgRbtNode>>::iterator sceneRbt = g_rbtNodes.begin();
    for(vector<RigTForm>::iterator c0 = g_currentKeyFrame->begin(), c00 = pprevFrame->begin(), 
        c1 = nextFrame->begin(), c11 = nnextFrame->begin(), end = g_currentKeyFrame->end(); 
        c0 != end; c0++, c1++, c00++, c11++) {
        RigTForm d = bezierInterpolate3(*c00, *c0, *c1, 1./6), 
            e = bezierInterpolate3(*c0, *c1, *c11, -1./6);
        (*sceneRbt) -> setRbt(catmullRom(c0,d,e,c1,alpha));
        sceneRbt++;
    }
    return false;
}

void endAnimation() {
    cout << "Finished playing animation..." << endl;
    g_playingAnimation = false;
    g_animateTime = 0;
    g_currentKeyFrame = g_keyFrames.end();
    g_currentKeyFrame--; g_currentKeyFrame--;
    g_currentKeyFrameNumber = g_keyFrames.size()-2;
    loadFrame();
    cout << "Now at frame [" << g_currentKeyFrameNumber << "]" << endl;
}

void animationUpdate() {
    if (g_playingAnimation) {
        bool endReached = interpolate((float) g_animateTime / g_msBetweenKeyFrames);
        if (!endReached)
            g_animateTime += 1000./g_framesPerSecond;
        else {
            endAnimation();
        }
    }
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

    if (g_shellNeedsUpdate) {
        updateShellGeometry();
    }

    if (!picking) {
        Drawer drawer(invEyeRbt, uniforms);
        g_world->accept(drawer);

        // draw spheres
        if (g_currentPickedRbtNode) {
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

        if (g_currentPickedRbtNode == g_groundNode)
            g_currentPickedRbtNode = shared_ptr<SgRbtNode>();   // set to NULL
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
    if (g_currentPickedRbtNode) { // manipulate object 
        RigTForm L = g_currentPickedRbtNode -> getRbt();
        RigTForm CL = getPathAccumRbt(g_world,g_currentPickedRbtNode);
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
        } else {
            L = doMtoOwrtA(m, L, As);
        }
        g_currentPickedRbtNode -> setRbt(L);
    } else if (g_activeEyeNode == g_skyNode) {
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
        g_skyNode -> setRbt(L);
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
            g_pickingMode = true;
            cout << "Picking mode is on" << endl;
            break;
        case GLFW_KEY_C:
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
            break;
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
                                "Assignment 9", NULL, NULL);
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
}

static void initMaterials() {
    // Create some prototype materials
    Material diffuse("./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader");
    Material solid("./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader");

    // copy diffuse prototype and set red color
    g_redDiffuseMat.reset(new Material(diffuse));
    g_redDiffuseMat->getUniforms().put("uColor", Cvec3f(165./255, 124./255, 0./255));

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

    // bunny material
    g_bunnyMat.reset(new Material("./shaders/basic-gl3.vshader",
                                  "./shaders/bunny-gl3.fshader"));
    g_bunnyMat->getUniforms().put("uColorAmbient", Cvec3f(0.45f, 0.3f, 0.3f))
                             .put("uColorDiffuse", Cvec3f(0.2f, 0.2f, 0.2f));

    // bunny shell materials;
    // common shell texture:
    shared_ptr<ImageTexture> shellTexture(new ImageTexture("shell.ppm", false));

    // enable repeating of texture coordinates
    shellTexture->bind();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // eachy layer of the shell uses a different material, though the materials
    // will share the same shader files and some common uniforms. hence we
    // create a prototype here, and will copy from the prototype later
    Material bunnyShellMatPrototype("./shaders/bunny-shell-gl3.vshader",
                                    "./shaders/bunny-shell-gl3.fshader");
    bunnyShellMatPrototype.getUniforms().put("uTexShell", shellTexture);
    bunnyShellMatPrototype.getRenderStates()
        .blendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) // set blending mode
        .enable(GL_BLEND)                                // enable blending
        .disable(GL_CULL_FACE);                          // disable culling

    // allocate array of materials
    g_bunnyShellMats.resize(g_numShells);
    for (int i = 0; i < g_numShells; ++i) {
        // copy prototype
        g_bunnyShellMats[i].reset(new Material(bunnyShellMatPrototype));
        // but set a different exponent for blending transparency
        g_bunnyShellMats[i]->getUniforms().put(
            "uAlphaExponent", 2.f + 5.f * float(i + 1) / g_numShells);
    }
};


static void initGeometry() {
    initGround();
    initCubes();
    initSphere();
    initBunnyMeshes();
}

static void constructRobot(shared_ptr<SgTransformNode> base, shared_ptr<Material> material) {

    const float ARM_LEN = 0.7, LEG_LEN = 1.0, NECK_LEN = 0.2, 
                ARM_THICK = 0.25, LEG_THICK = 0.35, 
                TORSO_LEN = 1.5, TORSO_THICK = 0.25, TORSO_WIDTH = 1,
                HEAD_LEN = 0.4, HEAD_THICK = 0.4, HEAD_WIDTH = 0.4;
    const int NUM_JOINTS = 10,
              NUM_SHAPES = 10;

    struct JointDesc {
        int parent;
        float x, y, z;
    };

    JointDesc jointDesc[NUM_JOINTS] = {
        {-1}, // torso
        {0,  TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper right arm
        {1,  ARM_LEN, 0, 0}, // lower right arm
        {0,  -TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper left arm
        {3,  -ARM_LEN, 0, 0}, // lower left arm
        {0,  0, NECK_LEN + TORSO_LEN/2, 0}, // head
        {0,  TORSO_WIDTH/4, -TORSO_LEN/2, 0}, // upper right leg
        {6,  0, -LEG_LEN, 0}, // lower right leg
        {0,  -TORSO_WIDTH/4, -TORSO_LEN/2, 0}, // upper left leg
        {8,  0, -LEG_LEN, 0}, // lower left leg
    };

    struct ShapeDesc {
        int parentJointId;
        float x, y, z, sx, sy, sz;
        shared_ptr<Geometry> geometry;
    };

    ShapeDesc shapeDesc[NUM_SHAPES] = {
        {0, 0,          0, 0, TORSO_WIDTH, TORSO_LEN, TORSO_THICK, g_cube}, // torso
        {1, ARM_LEN/2,  0, 0, ARM_LEN/2, ARM_THICK/2, ARM_THICK/2, g_sphere}, // upper right arm
        {2, ARM_LEN/2,  0, 0, ARM_LEN, ARM_THICK/2, ARM_THICK/2, g_cube}, // lower right arm
        {3, -ARM_LEN/2, 0, 0, ARM_LEN/2, ARM_THICK/2, ARM_THICK/2, g_sphere}, // upper right arm
        {4, -ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK/2, ARM_THICK/2, g_cube}, // lower right arm
        {5, 0, HEAD_LEN, 0, HEAD_WIDTH, HEAD_LEN, HEAD_THICK, g_sphere}, // head
        {6, 0, -LEG_LEN/2, 0, LEG_THICK/2, LEG_LEN/2, LEG_THICK/2, g_sphere}, // upper right leg
        {7, 0, -LEG_LEN/2, 0, LEG_THICK/2, LEG_LEN, LEG_THICK/2, g_cube}, // lower right leg
        {8, 0, -LEG_LEN/2, 0, LEG_THICK/2, LEG_LEN/2, LEG_THICK/2, g_sphere}, // upper left leg
        {9, 0, -LEG_LEN/2, 0, LEG_THICK/2, LEG_LEN, LEG_THICK/2, g_cube}, // lower left leg 
    };

    shared_ptr<SgTransformNode> jointNodes[NUM_JOINTS];

    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (jointDesc[i].parent == -1)
            jointNodes[i] = base;
        else {
            jointNodes[i].reset(new SgRbtNode(RigTForm(Cvec3(jointDesc[i].x, jointDesc[i].y, jointDesc[i].z))));
            jointNodes[jointDesc[i].parent]->addChild(jointNodes[i]);
        }
    }
    for (int i = 0; i < NUM_SHAPES; ++i) {
        shared_ptr<SgGeometryShapeNode> shape(
            new MyShapeNode(shapeDesc[i].geometry,
                            material, // USE MATERIAL as opposed to color
                            Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
                            Cvec3(0, 0, 0),
                            Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
        jointNodes[shapeDesc[i].parentJointId]->addChild(shape);
    }
}

static void initScene() {
    g_world.reset(new SgRootNode());

    g_skyNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, 0.25, 10.0))));
    g_activeEyeNode = g_skyNode;

    g_groundNode.reset(new SgRbtNode());
    g_groundNode->addChild(shared_ptr<MyShapeNode>(
                               new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(0, g_groundY, 0))));
    g_light1Node.reset(new SgRbtNode(RigTForm(Cvec3(6.0, 2.0, 12.0))));
    g_light1Node->addChild(shared_ptr<MyShapeNode>(
                               new MyShapeNode(g_sphere, g_lightMat, Cvec3(0, 0, 0))));

    g_light2Node.reset(new SgRbtNode(RigTForm(Cvec3(-3.0, 2.0, -15.0))));
    g_light2Node->addChild(shared_ptr<MyShapeNode>(
                               new MyShapeNode(g_sphere, g_lightMat, Cvec3(0, 0, 0))));

    g_robot1Node.reset(new SgRbtNode(RigTForm(Cvec3(-5, 1, 0))));
    g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(5, 1, 0))));

    constructRobot(g_robot1Node, g_redDiffuseMat); // a Red robot
    constructRobot(g_robot2Node, g_blueDiffuseMat); // a Blue robot

    // create a single transform node for both the bunny and the bunny
    // shells
    g_bunnyNode.reset(new SgRbtNode());

    // add bunny as a shape nodes
    g_bunnyNode->addChild(
        shared_ptr<MyShapeNode>(new MyShapeNode(g_bunnyGeometry, g_bunnyMat)));

    // add each shell as shape node
    for (int i = 0; i < g_numShells; ++i) {
        g_bunnyNode->addChild(shared_ptr<MyShapeNode>(
            new MyShapeNode(g_bunnyShellGeometries[i], g_bunnyShellMats[i])));
    }
    // from this point, calling g_bunnyShellGeometries[i]->upload(...) will
    // change the geometry of the ith layer of shell that gets drawn

    g_world->addChild(g_skyNode);
    g_world->addChild(g_groundNode);
    g_world->addChild(g_robot1Node);
    g_world->addChild(g_robot2Node);
    g_world->addChild(g_light1Node);
    g_world->addChild(g_light2Node);
    g_world->addChild(g_bunnyNode);
}

static void glfwLoop() {
    g_lastFrameClock = glfwGetTime();

    while (!glfwWindowShouldClose(g_window)) {
        double thisTime = glfwGetTime();
        if( thisTime - g_lastFrameClock >= 1. / g_framesPerSecond) {

            animationUpdate();
            hairsSimulationUpdate();
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
        initFrame();
        updateShellGeometry();

        glfwLoop();
        return 0;
    } catch (const runtime_error &e) {
        cout << "Exception caught: " << e.what() << endl;
        return -1;
    }
}
