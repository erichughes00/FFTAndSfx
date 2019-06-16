#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <tinyalsa/pcm.h>
#include <queue>
#include <thread>
#include <mutex>
#include <cmath>

#include "gpu_fft.h"
#include "mailbox.h"
#include "matlab/matplotlibcpp.h"
namespace plt = matplotlibcpp;

// test
const int fftSize = 12;
const int maxFFTReadSize = 4096;
const int sampleRate = 48000;
const float pi = 3.14159265358979323846;
int peakCount = 1;
int globalHertzModulation = 2; 


// Note: In order to end process incase the process doesn't end, enter sudo kill (the process ID (PID) in the command "top") 

// The class that holds all data about frames
class FrameData
{
	public:
	int frameCount;
	int byteCount;
	void* data;
	
	FrameData(int frameCount, int byteCount)
	{
		data = malloc(byteCount);
		this->frameCount = frameCount;
		this->byteCount = byteCount;
		if (data == NULL) {
			fprintf(stderr, "failed to allocate frames: read\n");
		}
	}
	
	// Move constructor
	FrameData(FrameData&& theOG)
	{
		frameCount = theOG.frameCount;
		byteCount = theOG.byteCount;
		data = theOG.data;
		theOG.data = nullptr;
	}
	
	FrameData(const FrameData& old_obj) = delete;
	
	~FrameData()
	{
		if (data != nullptr)
			free(data);
	}
};


class PulseCodeModulationBoi
{
	public:
	PulseCodeModulationBoi(int device, int card, int phlagz)
	{
		Open(device, card, phlagz);
	}
	
	~PulseCodeModulationBoi()
	{
		Close();
	}
	
	int Write(const FrameData& fData)
	{
		return pcm_writei(pcm, fData.data, fData.frameCount);
	}
	
	// Reads wh
	FrameData ReadingRainbow()
	{
		unsigned int frame_size = pcm_frames_to_bytes(pcm, 1);
		unsigned int frames_per_sec = pcm_get_rate(pcm);		
		int read_time = frames_per_sec / 10; // Read 10 milliseconds
		
		FrameData returnedFData(read_time, 
			frame_size * read_time);
			
		int read_count = 
			pcm_readi(pcm, f.data, read_time);
			
		size_t byte_count = pcm_frames_to_bytes(pcm, read_count);
		
		returnedFData.byteCount = byte_count;
		returnedFData.frameCount = read_count;
		
		return returnedFData;
	}
	
	private:
	// setting up the PCM
	struct pcm* pcm;
	void Open(int device, int card, int flags)
	{
		
		struct pcm_config config;
		config.channels = 1;
		config.rate = sampleRate; // Read 48000 points of data per second
		config.format = PCM_FORMAT_S16_LE;
		config.period_size = 1024;
		config.period_count = 2;
		config.start_threshold = 0;
		config.silence_threshold = 0;
		config.stop_threshold = 0;

	    pcm = pcm_open(card, device, flags, &config);
		if (pcm == NULL) {
			fprintf(stderr, "failed to allocate memory for PCM read lol\n");
			return;
		} else if (!pcm_is_ready(pcm)){
			fprintf(stderr, "failed to open PCM read lol\n %s", pcm_get_error(pcm));
			pcm_close(pcm);
			pcm = nullptr;
			return;
		}
	}
	
	void Close()
	{
		if (pcm != nullptr)
		{
			pcm_close(pcm);
		}
		
	}
};

// Class to perform audio processing in between the input and output queues
class SFX
{
public:

	// Turn pass through on or off
	// (output directly from input queue or perform FFT/SineWave first)
	bool setPassthrough(bool enable)
	{
		passthrough = enable;
	}

	// Constructor
	// Pass in reference to input data queue, output data queue, and their respective object locks
	SFX(std::queue<FrameData>* inputQQ, std::queue<FrameData>* outputQQ,
	 std::mutex* inputMucinex, std::mutex* outputMucinex)
	{
		inputQueue = inputQQ;
		outputQueue = outputQQ;
		inputMutex = inputMucinex;
		outputMutex = outputMucinex;
		mailbox = mbox_open();
		gpu_fft_prepare(mailbox, fftSize, GPU_FFT_FWD, 1, &fftInfo);
		
	}
	
	// Destructor
	~SFX()
	{
		gpu_fft_release(fftInfo);
	}
	
	// Begin FF-Transforming data from the data queue
	void BeginTransform()
	{
		while (true)
		{			
			(*inputMutex).lock();
			if (inputQueue->empty())
			{		
				(*inputMutex).unlock();	
			}
			else
			{
				FrameData fData = std::move(inputQueue->front());
				(*inputQueue).pop();
				(*inputMutex).unlock();	
				
				if(!passthrough)
				{
					SineWave(fData);	
				}
				std::lock_guard<std::mutex> lock(*outputMutex);
				outputQueue->push(std::move(fData));
			}

		}
	}
	
private:
	int mailbox;
	struct GPU_FFT* fftInfo;
	std::queue<FrameData>* inputQueue;
	std::queue<FrameData>* outputQueue;
	std::mutex* inputMutex;
	std::mutex* outputMutex;
	bool passthrough;
	int64_t phaseShift = 0;
	float prevSineX;
	
	const int hertzModulationMultiple = 2;
	const double baseNote = 27.5;
	const double halfStepConstant = 1.05946309536;
	
	void Modulate(std::vector<std::pair<float, float>>& dataReference)
	{
		//for(std::pair<float, float> frequency:(*dataReference)) // <Frequency, Amplitude>
		//{				
		//	frequency.first *= hertzModulationMultiple;
		//}
	}
	
	// Output added sine waves
	// based on samples in the form of FrameData
	void SineWave(FrameData& fData) // frequency to output
	{
		std::vector<std::pair<float, float>> frequencies = FastFourierTransform(fData);
		
		//for(std::pair<float, float> frequency:frequencies) // <Frequency, Amplitude>
		{				
		//	frequency.first *= hertzModulationMultiple;
		}
		
		int16_t* posZero = (int16_t*)fData.data;
		for (int i = 0; i < sampleRate / 10; i++)
		{
			int16_t amplitude = 0;
			for(std::pair<float, float> frequency:frequencies) // <Frequency, Amplitude>
			{				
				float frequencyFactor = globalHertzModulation * frequency.first * (2 * pi / sampleRate);
				prevSineX = std::sin((i + phaseShift) * frequencyFactor);
				amplitude += (int16_t)(prevSineX * frequency.second / 1000);
			}
			
			posZero[i] = amplitude;
		}
		// increment the Phaseshift variable to reduce clicking.
		phaseShift += sampleRate / 10;
	}
	
	// Perform the pythaogorean theorem given A and B
	float PythagoreanTheorem(float a, float b)
	{
		return std::sqrt((a * a) + (b * b));
	}
	
	// Perform a fast fourier transform on a sample of audio 
	// and return a vector containing a pair of <frequency, amplitude> 
	// representing the FFTransformed data
	std::vector<std::pair<float, float>> FastFourierTransform(FrameData& fData)
	{
		
		for (int i = 0; i < maxFFTReadSize; i++)
		{
			struct GPU_FFT_COMPLEX complexBoi;
			complexBoi.re = (float)(((int16_t*)fData.data)[i]);
			complexBoi.im = (float)0;
			(fftInfo->in)[i] = complexBoi;
		}
		// Preform the FFT and put data int fftInfo
		gpu_fft_execute(fftInfo);
		
		// we use a priority queue to order the freuqencies by amplitude, we only care about the loudest frequencies
		auto cmp = [](std::pair<float, float> ichi, std::pair<float, float> ni) { return ichi.second > ni.second; };
		std::priority_queue<std::pair<float, float>, std::vector<std::pair<float, float>>, decltype(cmp)> qq(cmp);
		
		// Initialize variables
		double bucketAmplitude = 0;
		int elementsInBucket = 0;
		int previousBucket = 0;
		for (int i = 0; i < maxFFTReadSize / 2; i++)
		{
			// The pathagoreanTheorem gives us the amplitude of given fft data.
			float currentAmplitude = PythagoreanTheorem((fftInfo->out)[i].re, (fftInfo->out)[i].im);
			float currentFrequency = (sampleRate * i) / maxFFTReadSize, currentBoi;
			
			// We don't care about extreamly high frequencies, so just end the loop at this point
			if (currentFrequency >= 4186)
				break;
			
			double distance = NumberOfHalfStepsFromANaturalZero(currentFrequency);
			
			// THe current bucket we are puting data into is the same as the distance from A0,
			// i.e. A0 is bucket 0, A#0 is bucket 2 B0 is bucket 3 etc...
			int currentBucket = std::round(distance);
			
			// Once we get data from a new bucket, finalize the previous bucket and start the new one
			if (previousBucket != currentBucket)
			{
				int finalAmplitude = bucketAmplitude / elementsInBucket;
				float finalFrequency = baseNote * std::pow(halfStepConstant, previousBucket);
				std::pair<float, float> currentPair = 
					std::make_pair(finalFrequency, finalAmplitude);
				qq.push(currentPair);
			
				if (qq.size() > peakCount)
					qq.pop();
				
				bucketAmplitude = 0;
				elementsInBucket = 0;
				previousBucket = currentBucket;
			}
			// Increment the amplitude and the number of elements in the current bucket.
			bucketAmplitude += currentAmplitude;
			elementsInBucket++;
		}
		
		// put all the data from the FFT into a vector of requested size in order to return
		// We used a priority queue to order our data, so returning only the data we want is easy.
		std::vector<std::pair<float, float>> data;
		for (int i = 0; i < peakCount; i++)
		{	
			data.push_back(qq.top());
			qq.pop();
		}
		
		return data;
	}
	
	// Calculate the number of half steps from A natural 0 in the western music scale
	// based on a frequency
	double NumberOfHalfStepsFromANaturalZero(double oldFrequency)
	{
		return (std::log(oldFrequency / baseNote)) / (std::log(halfStepConstant));
	}
};

// The input class to be threaded
class Input
{
	public:
	
	// Constructor
	// Pass in a reference to the data queue and the object lock
	Input(std::queue<FrameData>* qq, std::mutex* bm)
	{
		buffer = qq;
		buffer_mutex = bm;
	}
	
	// Begin reading infinitely
	void BeginRead()
	{
		ReadLimited(-1);
	}
	
	// Read a number of samples and put them into the buffer
	// -1 represents infinite reading
	void ReadLimited(int iterations)
	{
		PulseCodeModulationBoi pcmBoi(0, 1, PCM_IN);
		
		while (iterations == -1 || iterations-- > 0)
		{
			FrameData fData = pcmBoi.ReadingRainbow();
			std::lock_guard<std::mutex> lock(*buffer_mutex);
			//effectsBoi.PrintFFT(fData);
			buffer->push(std::move(fData));
		}
	}
	
	private:
	std::queue<FrameData>* buffer;
	std::mutex* buffer_mutex;
};

// The output class that we use threaded
class Output
{
	public:
	
	// Constructor
	// pass in pointer to the data queue and a thread lock
    Output(std::queue<FrameData>* qq, std::mutex* bm)
	{
		buffer = qq;
		buffer_mutex = bm;
	}
	
	// Begin writing from the data queue
	void BeginWrite()
	{
		PulseCodeModulationBoi pcmBoi(0, 0, PCM_OUT);
		while (true)
		{
						
			buffer_mutex->lock();	
			if (!buffer->empty())
			{
				FrameData data = std::move(buffer->front());
				buffer->pop();
				buffer_mutex->unlock();
				pcmBoi.Write(data);		
			}
			else
				buffer_mutex->unlock();
		}
	}
	
	private:
	std::queue<FrameData>* buffer;
	std::mutex* buffer_mutex;
};

int main(int argc, char **argv)
{
	// A dellay to reduce chopy playback
	const int bufferDelay = 2;
	
	// start all the threads to take input and play output
	std::mutex outputBuffer_mutex;
	std::mutex inputBuffer_mutex;
	std::queue<FrameData> inputBuffer;
	std::queue<FrameData> outputBuffer;
	Input inputBoi(&inputBuffer, &inputBuffer_mutex);
	Output outputBoi(&outputBuffer, &outputBuffer_mutex);
	SFX sfxBoi(&inputBuffer, &outputBuffer, &inputBuffer_mutex, &outputBuffer_mutex);
	
	inputBoi.ReadLimited(bufferDelay);
	std::thread input(&Input::BeginRead, std::ref(inputBoi));
	std::thread sfx(&SFX::BeginTransform, std::ref(sfxBoi));
	std::thread output(&Output::BeginWrite, std::ref(outputBoi));
	
	
	char choice;
	
	// The user menu
	while (true)
	{
		system("clear");
		
		std::cout << "(E)nable FFT" << std::endl << "(D)isable FFT" << 
		std::endl << "(S)ine wave count (currently " << peakCount << ")" <<
		std::endl << "Change the (M)odulation multiplier (currently " << globalHertzModulation << ") " 
		<< std::endl << "(Q)uit" << std::endl;
		
		std::cin >> choice;
		
		switch(choice)
		{
			case 'E':
			case 'e':
				sfxBoi.setPassthrough(false);
				break;
			case 'D':
			case 'd':
				sfxBoi.setPassthrough(true);
				break;
			case 'S':
			case 's':
				std::cout << "Enter the number of sine waves to add: ";
				std::cin >> peakCount;
				break;
			case 'M':
			case 'm':
				std::cout << "Enter the modulation multiple: ";
				std::cin >> globalHertzModulation;
			case 'Q':
			case 'q':
				return 0;
				break;
			default:
				break;
		}
	}
	
	input.join();
	sfx.join();
	output.join();
	
	
	return 0;
}
