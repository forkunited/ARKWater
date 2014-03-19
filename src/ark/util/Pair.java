package ark.util;

/**
 * 
 * @author Jesse Dodge
 * 
 */
public class Pair<F, S> {
    private final F first; //first member of pair
    private final S second; //second member of pair

    public Pair(F first, S second) {
        this.first = first;
        this.second = second;
    }

    public F getFirst() {
        return this.first;
    }

    public S getSecond() {
        return this.second;
    }
}