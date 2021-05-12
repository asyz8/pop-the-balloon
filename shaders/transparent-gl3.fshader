#version 150

uniform vec4 uColor;

out vec4 fragColor;

void main() {
    fragColor = vec4(uColor);
}
