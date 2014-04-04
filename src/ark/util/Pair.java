package ark.util;

public class Pair<F, S> {
    private F first; 
    private S second; 

    public Pair(F first, S second) {
        this.first = first;
        this.second = second;
    }

    public boolean setFirst(F first) {
    	this.first = first;
    	return true;
    }
    
    public boolean setSecond(S second) {
    	this.second = second;
    	return true;
    }
    
    public F getFirst() {
        return this.first;
    }

    public S getSecond() {
        return this.second;
    }
}