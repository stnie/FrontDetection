#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include "../PyToC/pyarrayCasts.hpp"
#include <pybind11/numpy.h>


// how to represent a deletion
#define DELETION_VAL -10000


constexpr inline size_t binom(size_t n, size_t k) noexcept
{
    return
      (        k> n  )? 0 :          // out of range
      (k==0 || k==n  )? 1 :          // edge
      (k==1 || k==n-1)? n :          // first
      (     k+k < n  )?              // recursive:
      (binom(n-1,k-1) * n)/k :       //  path to k=1   is faster
      (binom(n-1,k) * n)/(n-k);      //  path to k=n-1 is faster
}

float getGaussianValue(int x, int y, float sigma=1){
    return exp(-0.5 * (pow(x/sigma,2)+ pow(y/sigma,2)));
}

int getx(int i, int maxDist){
    int totalDist = 2*maxDist+1;
    i = i%totalDist;
    return i-maxDist; 
}

int gety(int i, int maxDist){
    int totalDist = 2*maxDist+1;
    i = i/totalDist;
    return i-maxDist;
}
template<class T>
float evalScore(int row, int col, float positionalPenalty, simpleMatrix<T>& image, int border){
    if(border<= row && row < image.x-border && border <= col && col < image.y-border){
        return (1+positionalPenalty)*image[row*image.y+col]-1;
    }
    return 0;
}

template<class T>
float evalScore(int row, int col, float positionalPenalty, simple3DMatrix<T>& image, int channel, int border){
    if(border<= row && row < image.x -border && border <= col && col < image.y-border){
        return (1+positionalPenalty)*image[row*image.y*image.z+col*image.z+channel]-1;
    }
    return 0;
}

template<class T>
void countPositivesMultiChannelInPlace(int row, int col, simple3DMatrix<T>& image, std::vector<float>& scores){
    for(int channel = 0;channel<image.z;++channel)
        scores[channel] += image[row*image.y*image.z+col*image.z+channel];
}


template<class T>
float countPositives(int row, int col, simpleMatrix<T>& image){
    return image[row*image.y+col];
}



template<class image_t>
float getScore(int r0, int c0, int r1, int c1, image_t& image, float positionalPenalty, int border){
    // get line pixel
    bool steep = false;
    int r = r0;
    int c = c0;
    int dr = abs(r1 - r0);
    int dc = abs(c1 - c0);
    int sr, sc, d, rr, cc;
    float score = 0;
    int validPoints=0;

    if((c1 - c) > 0)
        sc = 1;
    else
        sc = -1;
    if((r1 - r) > 0)
        sr = 1;
    else
        sr = -1;
    if(dr > dc){
        steep = true;
        std::swap(c,r);
        std::swap(dc,dr);
        std::swap(sc,sr);
    }
    d = (2 * dr) - dc;

    for(int i = 0;i< dc;++i){
        if(steep){
            rr = c;
            cc = r;
        }
        else{
            rr = r;
            cc = c;
        }
        if(border<= rr && rr < image.x-border && border <= cc && cc < image.y-border){
            score += countPositives(rr,cc, image);
            validPoints++;
        }
        while(d >= 0){
            r = r + sr;
            d = d - (2 * dc);
        }
        c = c + sc;
        d = d + (2 * dr);
    }

    rr = r1;
    cc = c1;
    if(border<= rr && rr < image.x-border && border <= cc && cc < image.y-border){
        score += countPositives(rr,cc, image);
        validPoints++;
    }
    //printf("%f\n", score);
    score = (1+positionalPenalty)*score - (validPoints);
    // get diff
    return score;
}

template<class image_t>
float getScoreBayes(int r0, int c0, int r1, int c1, image_t& image, float positionalPenalty, float positiveProbability, int border){
    // get line pixel
    bool steep = false;
    int r = r0;
    int c = c0;
    int dr = abs(r1 - r0);
    int dc = abs(c1 - c0);
    int sr, sc, d, rr, cc;
    float score = 0;
    int validPoints = 0;

    if((c1 - c) > 0)
        sc = 1;
    else
        sc = -1;
    if((r1 - r) > 0)
        sr = 1;
    else
        sr = -1;
    if(dr > dc){
        steep = true;
        std::swap(c,r);
        std::swap(dc,dr);
        std::swap(sc,sr);
    }
    d = (2 * dr) - dc;

    for(int i = 0;i< dc;++i){
        if(steep){
            rr = c;
            cc = r;
        }
        else{
            rr = r;
            cc = c;
        }
        if(border<= rr && rr < image.x-border && border <= cc && cc < image.y-border){
            score += countPositives(rr,cc, image);
            validPoints++;
        }
        while(d >= 0){
            r = r + sr;
            d = d - (2 * dc);
        }
        c = c + sc;
        d = d + (2 * dr);
    }

    rr = r1;
    cc = c1;
    if(border<= rr && rr < image.x-border && border <= cc && cc < image.y-border){
        score += countPositives(rr,cc, image);
        validPoints++;
    }
    // We got the number of positive results in score and the number of total points in dc
    int pA = positionalPenalty;
    float pBA = binom(validPoints, score) * pow(positiveProbability, score) * pow(1-positiveProbability, validPoints-score);
    float condProb = pBA * pA; 

    // get diff
    return condProb;
}




template<class T>
std::pair<float,int> getScoreAndChannel(int r0, int c0, int r1, int c1, simple3DMatrix<T>& image, float positionalPenalty, simpleVector<float>& channelPenalties, int border){
    // get line pixel
    float bestScore = -10000;
    int bestChannel = -1;
    std::vector<float> scores(image.z);
    bool steep = false;
    int r = r0;
    int c = c0;
    int dr = abs(r1 - r0);
    int dc = abs(c1 - c0);
    int sr, sc, d, rr, cc;
    int validPoints = 0;
    

    if((c1 - c) > 0)
        sc = 1;
    else
        sc = -1;
    if((r1 - r) > 0)
        sr = 1;
    else
        sr = -1;
    if(dr > dc){
        steep = true;
        std::swap(c,r);
        std::swap(dc,dr);
        std::swap(sc,sr);
    }
    d = (2 * dr) - dc;

    for(int i = 0;i< dc;++i){
        if(steep){
            rr = c;
            cc = r;
        }
        else{
            rr = r;
            cc = c;
        }
        if(border<= rr && rr < image.x-border && border <= cc && cc < image.y-border){
            countPositivesMultiChannelInPlace(rr,cc, image,scores);
            validPoints++;
        }
        while(d >= 0){
            r = r + sr;
            d = d - (2 * dc);
        }
        c = c + sc;
        d = d + (2 * dr);
    }

    rr = r1;
    cc = c1;
    if(border<= rr && rr < image.x-border && border <= cc && cc < image.y-border){
        countPositivesMultiChannelInPlace(rr,cc, image,scores);
        validPoints++;
    }
    for(int i = 0;i<image.z;++i){
        scores[i] = scores[i]*(1+positionalPenalty*channelPenalties[i])-validPoints;
        if(scores[i] > bestScore){
            bestScore = scores[i];
            bestChannel = i;
        }
    }

    // get diff
    return std::pair<float,int>(bestScore, bestChannel);
}




template<class T>
simpleMatrix<int> getBestPath_(simpleMatrix<int>&& coords, simpleMatrix<T>&& image, int maxDist, float sigma, float deletion_err , int border){
    int totalDist = 2*maxDist+1;
    // Vector that holds the best origin for each explored point
    // totalDist * totalDist = exploration range for each vertex
    // coords.size = number of vertices * number of ints per coord (2d, 3d, etc)
    // NEW VERSION: Add another field that represents deletion of the node
    std::vector<int> bestOrigin((totalDist*totalDist+1)*coords.size());
    std::vector<float> bestScore((totalDist*totalDist+1)*coords.x);
    
    int oldLat = coords[0];
    int oldLon = coords[1];
    float positionalPenalty = 1;
    for(int i = 0;i< totalDist*totalDist;++i){
        positionalPenalty = getGaussianValue(getx(i,maxDist), gety(i,maxDist), sigma);
        bestScore[i] = getScore(oldLat,oldLon,oldLat,oldLon,image, positionalPenalty, border);
        
    }
    bestScore[totalDist*totalDist] = deletion_err;
    int lat,lon,sourceLat, sourceLon, targetLat, targetLon;
    float score = -1;
    float myBestScore = -10000;
    int myBestOrigin = -1;
    // for each coordinate pair
    for(int c_idx = 1; c_idx < coords.x; ++c_idx){
        lat = coords[c_idx*coords.y];
        lon = coords[c_idx*coords.y+1];
        // for each offset of the target
        for(int off = 0; off < totalDist*totalDist; ++off){
            targetLat = lat+gety(off,maxDist);
            targetLon = lon+getx(off,maxDist);
            myBestScore = -10000;
            myBestOrigin = -1;
            // for each offset of the source
            for(int o_off = 0; o_off < totalDist*totalDist; ++o_off){
                sourceLat = oldLat+gety(o_off,maxDist);
                sourceLon = oldLon+getx(o_off,maxDist);
                positionalPenalty = getGaussianValue(getx(off,maxDist),gety(off,maxDist), sigma);
                score = getScore(sourceLat, sourceLon, targetLat, targetLon, image,positionalPenalty, border);
                
                score += bestScore[o_off+(totalDist*totalDist+1)*(c_idx-1)];
                if(score > myBestScore){
                    myBestScore = score;
                    myBestOrigin = o_off;
                }
            }
            // Check if a deletion of the source was favourable instead
            // deviation penalty of my new point
            positionalPenalty = getGaussianValue(getx(off,maxDist),gety(off,maxDist), sigma);
            // penalty of my new point
            score = getScore(targetLat, targetLon, targetLat, targetLon, image,positionalPenalty, border);
            // penalty for the deleted region
            float pixelInDist = std::max(std::max(abs(targetLat-oldLat), abs(targetLon-oldLon)),2);
            score += deletion_err*pixelInDist;

            score += bestScore[totalDist*totalDist+ (totalDist*totalDist+1)*(c_idx-1)];
            if(score > myBestScore){
                myBestScore = score;
                myBestOrigin = totalDist*totalDist;
            }

            bestScore[off+(totalDist*totalDist+1)*c_idx] = myBestScore;
            bestOrigin[off+(totalDist*totalDist+1)*c_idx] = myBestOrigin;
        }
        // Check if a deletion of the target is favourable instead
        myBestScore = -10000;
        myBestOrigin = -1;
        
        for(int o_off = 0; o_off < totalDist*totalDist; ++o_off){
            sourceLat = oldLat+gety(o_off,maxDist);
            sourceLon = oldLon+getx(o_off,maxDist);
            float pixelInDist = std::max(std::max(abs(lat-sourceLat), abs(lon-sourceLon)),2);
            score = deletion_err*pixelInDist;
            score += bestScore[o_off+(totalDist*totalDist+1)*(c_idx-1)];
            if(score > myBestScore){
                myBestScore = score;
                myBestOrigin = o_off;
            }
        }
        // check if a deletion of both the source and the target were optimal
        sourceLat = oldLat;
        sourceLon = oldLon;
        float pixelInDist = std::max(std::max(abs(lat-sourceLat), abs(lon-sourceLon)),2);
        score = deletion_err*pixelInDist;
        score += bestScore[totalDist*totalDist+(totalDist*totalDist+1)*(c_idx-1)];
        if(score > myBestScore){
            myBestScore = score;
            myBestOrigin = totalDist*totalDist;
        }

        bestScore[totalDist*totalDist+(totalDist*totalDist+1)*c_idx] = myBestScore;
        bestOrigin[totalDist*totalDist+(totalDist*totalDist+1)*c_idx] = myBestOrigin;

        oldLat = lat;
        oldLon = lon;
    }

    int bestEnd = 0;
    float bestEndScore = 0;

    
    for(int off = 0; off < totalDist*totalDist+1; ++off){
        if(bestScore[(totalDist*totalDist+1)*(coords.x-1)+off] > bestEndScore){
            bestEnd = off;
            bestEndScore = bestScore[(totalDist*totalDist+1)*(coords.x-1)+off];
        }
    }
    // get best Path and return it
    for(int c_idx = coords.x-1; c_idx >= 0; --c_idx){
        // in case of deletion, write a distinct value
        if(bestEnd == totalDist*totalDist){
            coords.data[c_idx*coords.y] = DELETION_VAL;
            coords.data[c_idx*coords.y+1] = DELETION_VAL;
        }
        else{
            coords.data[c_idx*coords.y] += gety(bestEnd, maxDist);
            coords.data[c_idx*coords.y+1] += getx(bestEnd, maxDist);
        }
        bestEnd = bestOrigin[c_idx*(totalDist*totalDist+1)+bestEnd];
    }
    return coords;
}


template<class T>
simpleMatrix<int> getBestPathBayes_(simpleMatrix<int>&& coords, simpleMatrix<T>&& image, int maxDist, float sigma, float positiveProbability, float deletion_err, int border){
    int totalDist = 2*maxDist+1;
    // Vector that holds the best origin for each explored point
    // totalDist * totalDist = exploration range for each vertex
    // coords.size = number of vertices * number of ints per coord (2d, 3d, etc)
    // NEW VERSION: Add another field that represents deletion of the node
    std::vector<int> bestOrigin((totalDist*totalDist+1)*coords.size());
    std::vector<float> bestScore((totalDist*totalDist+1)*coords.x);
    
    int oldLat = coords[0];
    int oldLon = coords[1];
    float positionalPenalty = 1;
    for(int i = 0;i< totalDist*totalDist;++i){
        positionalPenalty = getGaussianValue(getx(i,maxDist), gety(i,maxDist), sigma);
        bestScore[i] = getScoreBayes(oldLat,oldLon,oldLat,oldLon,image, positionalPenalty, positiveProbability, border);
        
    }
    bestScore[totalDist*totalDist] = deletion_err;
    int lat,lon,sourceLat, sourceLon, targetLat, targetLon;
    float score = -1;
    float myBestScore = -10000;
    int myBestOrigin = -1;
    // for each coordinate pair
    for(int c_idx = 1; c_idx < coords.x; ++c_idx){
        lat = coords[c_idx*coords.y];
        lon = coords[c_idx*coords.y+1];
        // for each offset of the target
        for(int off = 0; off < totalDist*totalDist; ++off){
            targetLat = lat+gety(off,maxDist);
            targetLon = lon+getx(off,maxDist);
            myBestScore = -10000;
            myBestOrigin = -1;
            // for each offset of the source
            for(int o_off = 0; o_off < totalDist*totalDist; ++o_off){
                sourceLat = oldLat+gety(o_off,maxDist);
                sourceLon = oldLon+getx(o_off,maxDist);
                positionalPenalty = getGaussianValue(getx(off,maxDist),gety(off,maxDist), sigma);
                score = getScoreBayes(sourceLat, sourceLon, targetLat, targetLon, image,positionalPenalty, positiveProbability, border);
                
                score += bestScore[o_off+(totalDist*totalDist+1)*(c_idx-1)];
                if(score > myBestScore){
                    myBestScore = score;
                    myBestOrigin = o_off;
                }
            }
            // Check if a deletion of the source was favourable instead
            positionalPenalty = getGaussianValue(getx(off,maxDist),gety(off,maxDist), sigma);
            score = getScoreBayes(targetLat, targetLon, targetLat, targetLon, image,positionalPenalty, positiveProbability, border);
            score += bestScore[totalDist*totalDist+ (totalDist*totalDist+1)*(c_idx-1)];
            if(score > myBestScore){
                myBestScore = score;
                myBestOrigin = totalDist*totalDist;
            }

            bestScore[off+(totalDist*totalDist+1)*c_idx] = myBestScore;
            bestOrigin[off+(totalDist*totalDist+1)*c_idx] = myBestOrigin;
        }
        // Check if a deletion of the target is favourable instead
        myBestScore = -10000;
        myBestOrigin = -1;
        float pixelInDist = std::max(abs(lat-oldLat), abs(lon-oldLon));
        for(int o_off = 0; o_off < totalDist*totalDist+1; ++o_off){
            score = deletion_err*pixelInDist;
            score += bestScore[o_off+(totalDist*totalDist+1)*(c_idx-1)];
            if(score > myBestScore){
                myBestScore = score;
                myBestOrigin = o_off;
            }
        }

        bestScore[totalDist*totalDist+(totalDist*totalDist+1)*c_idx] = myBestScore;
        bestOrigin[totalDist*totalDist+(totalDist*totalDist+1)*c_idx] = myBestOrigin;

        oldLat = lat;
        oldLon = lon;
    }

    int bestEnd = 0;
    float bestEndScore = 0;

    
    for(int off = 0; off < totalDist*totalDist+1; ++off){
        if(bestScore[(totalDist*totalDist+1)*(coords.x-1)+off] > bestEndScore){
            bestEnd = off;
            bestEndScore = bestScore[(totalDist*totalDist+1)*(coords.x-1)+off];
        }
    }
    // get best Path and return it
    for(int c_idx = coords.x-1; c_idx >= 0; --c_idx){
        // in case of deletion, write a distinct value
        if(bestEnd == totalDist*totalDist){
            coords.data[c_idx*coords.y] = DELETION_VAL;
            coords.data[c_idx*coords.y+1] = DELETION_VAL;
        }
        else{
            coords.data[c_idx*coords.y] += gety(bestEnd, maxDist);
            coords.data[c_idx*coords.y+1] += getx(bestEnd, maxDist);
        }
        bestEnd = bestOrigin[c_idx*(totalDist*totalDist+1)+bestEnd];
    }
    return coords;
}



template<class T>
simpleMatrix<int> getBestPathMultiChannel_(simpleMatrix<int>&& coords, simple3DMatrix<T>&& image, int maxDist, float sigma, simpleVector<float>&& channelScore, float deletion_err, int border){
    int totalDist = 2*maxDist+1;
    // Vector that holds the best origin for each explored point
    // totalDist * totalDist = exploration range for each vertex
    // coords.size = number of vertices * number of ints per coord (2d, 3d, etc)
    // NEW VERSION: Add another field that represents deletion of the node

    std::vector<int> bestOrigin((totalDist*totalDist+1)*coords.size());
    std::vector<float> bestScore((totalDist*totalDist+1)*coords.x);
    std::vector<int> bestColor((totalDist*totalDist+1)*coords.x);
    
    int oldLat = coords[0];
    int oldLon = coords[1];
    float positionalPenalty = 1;
    for(int i = 0;i< totalDist*totalDist;++i){
        positionalPenalty = getGaussianValue(getx(i,maxDist), gety(i,maxDist), sigma);
        bestScore[i] = getScoreAndChannel(oldLat,oldLon,oldLat,oldLon,image, positionalPenalty, channelScore, border).first;
        
    }
    bestScore[totalDist*totalDist] = deletion_err;
    int lat,lon,sourceLat, sourceLon, targetLat, targetLon;
    float score = -1;
    std::pair<float, int> score_channel;
    float myBestScore = -10000;
    int myBestColor = -1;
    int myBestOrigin = -1;
    
    // for each coordinate pair
    for(int c_idx = 1; c_idx < coords.x; ++c_idx){
        lat = coords[c_idx*coords.y];
        lon = coords[c_idx*coords.y+1];
        // for each offset of the target
        for(int off = 0; off < totalDist*totalDist; ++off){
            targetLat = lat+gety(off,maxDist);
            targetLon = lon+getx(off,maxDist);
            myBestScore = -10000;
            myBestOrigin = -1;
            myBestColor = -1;
            // for each offset of the source
            for(int o_off = 0; o_off < totalDist*totalDist; ++o_off){
                sourceLat = oldLat+gety(o_off,maxDist);
                sourceLon = oldLon+getx(o_off,maxDist);
                positionalPenalty = getGaussianValue(getx(off,maxDist),gety(off,maxDist), sigma);
                score_channel = getScoreAndChannel(sourceLat, sourceLon, targetLat, targetLon, image, positionalPenalty, channelScore, border);
                score = score_channel.first;
                score += bestScore[o_off+(totalDist*totalDist+1)*(c_idx-1)];
                if(score > myBestScore){
                    myBestScore = score;
                    myBestOrigin = o_off;
                    myBestColor = score_channel.second;
                }
            }
            // Check if a deletion of the source was favourable instead
            positionalPenalty = getGaussianValue(getx(off,maxDist),gety(off,maxDist), sigma);
            score_channel = getScoreAndChannel(targetLat, targetLon, targetLat, targetLon, image, positionalPenalty, channelScore, border);
            score = score_channel.first;
            score += bestScore[totalDist*totalDist+ (totalDist*totalDist+1)*(c_idx-1)];
            if(score > myBestScore){
                myBestScore = score;
                myBestOrigin = totalDist*totalDist;
                myBestColor = score_channel.second;
            }

            bestScore[off+(totalDist*totalDist+1)*c_idx] = myBestScore;
            bestOrigin[off+(totalDist*totalDist+1)*c_idx] = myBestOrigin;
            bestColor[off+(totalDist*totalDist+1)*c_idx] = myBestColor;
        }
        // Check if a deletion of the target is favourable instead
        myBestScore = -10000;
        myBestOrigin = -1;
        myBestColor = -1;
        float pixelInDist = std::max(abs(lat-oldLat), abs(lon-oldLon));
        for(int o_off = 0; o_off < totalDist*totalDist+1; ++o_off){
            score = deletion_err*pixelInDist;
            score += bestScore[o_off+(totalDist*totalDist+1)*(c_idx-1)];
            if(score > myBestScore){
                myBestScore = score;
                myBestOrigin = o_off;
                myBestColor = 0;
            }
        }

        bestScore[totalDist*totalDist+(totalDist*totalDist+1)*c_idx] = myBestScore;
        bestOrigin[totalDist*totalDist+(totalDist*totalDist+1)*c_idx] = myBestOrigin;
        bestColor[totalDist*totalDist+(totalDist*totalDist+1)*c_idx] = myBestColor;
        
        oldLat = lat;
        oldLon = lon;
    }

    int bestEnd = 0;
    float bestEndScore = -10000;

    
    for(int off = 0; off < totalDist*totalDist+1; ++off){
        if(bestScore[(totalDist*totalDist+1)*(coords.x-1)+off] > bestEndScore){
            bestEnd = off;
            bestEndScore = bestScore[(totalDist*totalDist+1)*(coords.x-1)+off];
        }
    }
    // get best Path and return it
    int * returnData = new int[coords.x*(coords.y+1)];
    simpleMatrix<int> returnCoords{returnData, coords.x, coords.y+1};
    for(int c_idx = coords.x-1; c_idx >= 0; --c_idx){
        // in case of deletion, write a distinct value
        if(bestEnd == totalDist*totalDist){
            returnCoords.data[c_idx*returnCoords.y] = DELETION_VAL;
            returnCoords.data[c_idx*returnCoords.y+1] = DELETION_VAL;
            returnCoords.data[c_idx*returnCoords.y+2] = DELETION_VAL;
        }
        else{
            returnCoords.data[c_idx*returnCoords.y] = coords.data[c_idx*coords.y]+gety(bestEnd, maxDist);
            returnCoords.data[c_idx*returnCoords.y+1] = coords.data[c_idx*coords.y+1]+getx(bestEnd, maxDist);
            returnCoords.data[c_idx*returnCoords.y+2] = bestColor[c_idx*(totalDist*totalDist+1)+bestEnd];
        }
        bestEnd = bestOrigin[c_idx*(totalDist*totalDist+1)+bestEnd];
    }
    return returnCoords;
}


pybind11::array_t<int , CMASK_ > getBestPath(pybind11::array_t<int , CMASK_ >& coords, pybind11::array_t<float , CMASK_ >& image, int maxDist, float sigma, float deletion_err = -5, int border = 0){
     return vecToPy2D(std::move(getBestPath_(pyToVec2D(coords), pyToVec2D(image), maxDist, sigma, deletion_err, border)));
}

pybind11::array_t<int , CMASK_ > getBestPathBayes(pybind11::array_t<int , CMASK_ >& coords, pybind11::array_t<float , CMASK_ >& image, int maxDist, float sigma, float positiveProbability, float deletion_err = -5, int border = 0){
     return vecToPy2D(std::move(getBestPathBayes_(pyToVec2D(coords), pyToVec2D(image), maxDist, sigma, positiveProbability, deletion_err, border)));
}

pybind11::array_t<int , CMASK_ > getBestPathAndChannel(pybind11::array_t<int , CMASK_ >& coords, pybind11::array_t<float , CMASK_ >& image, int maxDist, float sigma, pybind11::array_t<float, CMASK_>& channelScore, float deletion_err = -5, int border = 0){
     return vecToPy2D(std::move(getBestPathMultiChannel_(pyToVec2D(coords), pyToVec3D(image), maxDist, sigma, pyToVec1D(channelScore), deletion_err, border)));
}
